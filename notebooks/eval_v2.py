"""v2 evaluation: train on v2 features, then on the FULL val set compare
  - v1 vs v2 (threshold)                      overall / per country
  - crowded-address slice (France-like)       S1 sharing its exact address with other S1s or 4+ pool names there
  - decision strategies: threshold vs expected-F0.5 vs hybrid (+ temperature)
  - name-floor rule cost
"""
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code" / "business_entity_resolution"))
from src.decision import expected_f, threshold  # noqa: E402
from src.evaluate import evaluate  # noqa: E402
from src.train import PARAMS, feature_names, to_pairs  # noqa: E402

A = ROOT / "artifacts"
t0 = time.time()
split = pl.read_parquet(A / "splits/split_v1.parquet").with_columns(pl.col("s1_id").str.slice(3).cast(pl.UInt32).alias("s1"))
truth = pl.read_parquet(A / "processed/train_gt_pairs.parquet")
val_ids = split.filter(pl.col("role") == "val").select("s1_id", "country", "s1")

# ---------------- train v2
fit = pl.read_parquet(A / "features/fit_sample_v2/*.parquet")
dev = pl.read_parquet(A / "features/dev_v2/*.parquet")
feats = feature_names(fit)
X = lambda d: d.select([pl.col(c).cast(pl.Float32) for c in feats]).to_numpy()  # noqa: E731
dtr = lgb.Dataset(X(fit), fit["label"].to_numpy(), feature_name=feats)
ddv = lgb.Dataset(X(dev), dev["label"].to_numpy(), reference=dtr)
del fit
b = lgb.train(PARAMS, dtr, 3000, valid_sets=[ddv], callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(200)])
(A / "models").mkdir(exist_ok=True)
b.save_model(str(A / "models/lgb_v2.txt"))
print(f"v2 trained: {b.best_iteration} rounds, {len(feats)} features ({time.time()-t0:.0f}s)")
imp = sorted(zip(feats, b.feature_importance("gain")), key=lambda x: -x[1])
tot = sum(g for _, g in imp)
print("gain:", [(f, round(g / tot, 3)) for f, g in imp[:15]])

# ---------------- predict full val (keep name features for the name-floor analysis)
keep = ["s1", "m", "src", "label", "n_best", "n_conflict", "crowd_s1_n_1", "crowd_pool_names_1"]
parts = []
for p in sorted((A / "features/val_v2").glob("*.parquet")):
    d = pl.read_parquet(p)
    parts.append(d.select(keep).with_columns(pl.Series("p", b.predict(X(d), num_iteration=b.best_iteration), dtype=pl.Float32))
                  .filter(pl.col("p") >= 0.01))
pv = pl.concat(parts)
print(f"val predicted: {pv.height:,} pairs ({time.time()-t0:.0f}s)")

crowd = (pl.read_parquet(A / "features/val_v2/*.parquet", columns=["s1", "crowd_s1_n_1", "crowd_pool_names_1"])
           .group_by("s1").agg(pl.col("crowd_s1_n_1").max(), pl.col("crowd_pool_names_1").max()))
crowded = val_ids.join(crowd, on="s1", how="left").filter((pl.col("crowd_s1_n_1") >= 2) | (pl.col("crowd_pool_names_1") >= 4))
print(f"crowded slice: {crowded.height:,} of {val_ids.height:,} val S1 ({crowded.height / val_ids.height:.1%})")


def rep(name, sel):
    r = evaluate(to_pairs(sel), truth, val_ids.select("s1_id", "country"))
    rc = evaluate(to_pairs(sel), truth, crowded.select("s1_id", "country"))
    print(f"  {name:38s} val {r['f05']:.4f} (P {r['macro_p']:.4f} R {r['macro_r']:.4f} sing {r['singleton_acc']:.4f}) "
          f"US {r['by_country']['US']['f05']:.4f} IN {r['by_country']['India']['f05']:.4f} | crowded {rc['f05']:.4f}")
    return r["f05"]


print("\n[threshold]")
best_t = max(((t, rep(f"threshold {t:.2f}", threshold(pv, t))) for t in np.round(np.arange(0.5, 0.91, 0.05), 2)), key=lambda x: x[1])
print("\n[expected-F0.5]")
for temp in (0.8, 1.0, 1.25, 1.5):
    rep(f"expected_f temp {temp}", expected_f(pv, temp=temp))
for floor in (0.2, 0.3, 0.4):
    rep(f"hybrid floor {floor}", expected_f(pv, floor=floor))
print("\n[name floor on best threshold]")
for nf in (40, 50, 60):
    rep(f"thr {best_t[0]} + reject n_best<{nf}", threshold(pv, best_t[0]).filter(pl.col("n_best") >= nf))
print(f"done ({time.time()-t0:.0f}s)")
json.dump({"features": feats, "best_iteration": b.best_iteration, "threshold": best_t[0]},
          open(A / "models/lgb_v2.json", "w"), indent=1)
