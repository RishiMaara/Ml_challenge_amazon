"""Honest checks for v1 (Step 8.3/8.5):
  A. v1 on the FULL val set (441k S1): at the dev threshold and at a val-tuned threshold
  B. Leave-one-country-out: train on one country's labels, tune the threshold on that country's val, apply blindly
     to the other country (the France scenario), compared with v1 in-domain on the same S1s.
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
from src.evaluate import evaluate  # noqa: E402
from src.train import PARAMS, one_owner, to_pairs  # noqa: E402

A = ROOT / "artifacts"
GRID = np.round(np.arange(0.30, 0.91, 0.05), 2)
t0 = time.time()
split = pl.read_parquet(A / "splits/split_v1.parquet").with_columns(pl.col("s1_id").str.slice(3).cast(pl.UInt32).alias("s1"))
country_of = split.select("s1", "country")
truth = pl.read_parquet(A / "processed/train_gt_pairs.parquet")
val_ids = split.filter(pl.col("role") == "val").select("s1_id", "country")
meta = json.loads((A / "models/lgb_v1.json").read_text())
feats = meta["features"]
v1 = lgb.Booster(model_file=str(A / "models/lgb_v1.txt"))


def predict_parts(booster, feat_dir, keep_label=True):
    out = []
    for p in sorted(Path(feat_dir).glob("*.parquet")):
        df = pl.read_parquet(p)
        X = df.select([pl.col(c).cast(pl.Float32) for c in feats]).to_numpy()
        cols = ["s1", "m", "src"] + (["label"] if keep_label else [])
        out.append(df.select(cols).with_columns(pl.Series("p", booster.predict(X), dtype=pl.Float32)).filter(pl.col("p") >= 0.02))
    return pl.concat(out).join(country_of, on="s1")


def score(pred, ids, thr):
    return evaluate(to_pairs(one_owner(pred).filter(pl.col("p") >= thr)), truth, ids)


def sweep(pred, ids):
    owned = one_owner(pred)
    res = [(float(t), evaluate(to_pairs(owned.filter(pl.col("p") >= t)), truth, ids)["f05"]) for t in GRID]
    return max(res, key=lambda x: x[1]), res


# ---------------- A. full val
pv = predict_parts(v1, A / "features/val")
print(f"[A] val predictions: {pv.height:,} pairs ({time.time()-t0:.0f}s)")
r = score(pv, val_ids, meta["threshold"])
print(f"[A] v1 @ dev thr {meta['threshold']}: F0.5 {r['f05']:.4f} P {r['macro_p']:.4f} R {r['macro_r']:.4f} singleton {r['singleton_acc']:.4f}")
for c, v in r["by_country"].items():
    print(f"      {c}: F0.5 {v['f05']:.4f} P {v['macro_p']:.4f} R {v['macro_r']:.4f}")
(best_t, best_f), curve = sweep(pv, val_ids)
print(f"[A] val-tuned thr {best_t}: F0.5 {best_f:.4f} | curve " + " ".join(f"{t}:{f:.4f}" for t, f in curve))

# ---------------- B. LOCO
fit = pl.read_parquet(A / "features/fit_sample/*.parquet").join(country_of, on="s1")
dev = pl.read_parquet(A / "features/dev/*.parquet").join(country_of, on="s1")
for train_c, test_c in [("US", "India"), ("India", "US")]:
    f_tr, f_dev = fit.filter(pl.col("country") == train_c), dev.filter(pl.col("country") == train_c)
    dtr = lgb.Dataset(f_tr.select([pl.col(c).cast(pl.Float32) for c in feats]).to_numpy(), f_tr["label"].to_numpy(), feature_name=feats)
    ddv = lgb.Dataset(f_dev.select([pl.col(c).cast(pl.Float32) for c in feats]).to_numpy(), f_dev["label"].to_numpy(), reference=dtr)
    b = lgb.train(PARAMS, dtr, 2000, valid_sets=[ddv], callbacks=[lgb.early_stopping(100, verbose=False)])
    p = predict_parts(b, A / "features/val")
    ids_tr, ids_te = val_ids.filter(pl.col("country") == train_c), val_ids.filter(pl.col("country") == test_c)
    (t_src, f_src), _ = sweep(p.filter(pl.col("country") == train_c), ids_tr)
    blind = score(p.filter(pl.col("country") == test_c), ids_te, t_src)["f05"]
    (t_or, f_or), _ = sweep(p.filter(pl.col("country") == test_c), ids_te)
    in_dom = score(pv.filter(pl.col("country") == test_c), ids_te, best_t)["f05"]
    print(f"[B] train {train_c} ({b.best_iteration} rounds) → {test_c}: in-source {f_src:.4f} @thr {t_src} | "
          f"BLIND on {test_c} {blind:.4f} | {test_c}-tuned thr {t_or} → {f_or:.4f} | v1 in-domain {in_dom:.4f} "
          f"| unseen-country cost {in_dom - blind:+.4f} ({time.time()-t0:.0f}s)")
