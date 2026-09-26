"""Experiment: does stage-2 group consistency beat stage-1 alone? (US/India val, crowd features removed)

  stage-1  = LightGBM on v2 features minus crowd_* (trained on fit_sample_v2, early-stopped on dev_v2)
  stage-2  = stage-1 features + p1 + group features, trained on val half A, evaluated on val half B
  baseline = stage-1 + one-owner + best threshold on half B
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
from src.features import CROWD_COLS  # noqa: E402
from src.stage2 import GROUP_COLS, build  # noqa: E402
from src.train import PARAMS, feature_names, one_owner, to_pairs  # noqa: E402

A = ROOT / "artifacts"
t0 = time.time()
split = pl.read_parquet(A / "splits/split_v1.parquet").with_columns(pl.col("s1_id").str.slice(3).cast(pl.UInt32).alias("s1"))
truth = pl.read_parquet(A / "processed/train_gt_pairs.parquet")
X = lambda d, f: d.select([pl.col(c).cast(pl.Float32) for c in f]).to_numpy()  # noqa: E731

# ---------------- stage 1 (no crowd features)
fit = pl.read_parquet(A / "features/fit_sample_v2/*.parquet")
dev = pl.read_parquet(A / "features/dev_v2/*.parquet")
f1 = [c for c in feature_names(fit) if c not in CROWD_COLS]
b1 = lgb.train(PARAMS, lgb.Dataset(X(fit, f1), fit["label"].to_numpy(), feature_name=f1), 3000,
               valid_sets=[lgb.Dataset(X(dev, f1), dev["label"].to_numpy())], callbacks=[lgb.early_stopping(100, verbose=False)])
del fit
b1.save_model(str(A / "models/lgb_s1.txt"))
(A / "models/lgb_s1.json").write_text(json.dumps({"features": f1, "best_iteration": b1.best_iteration}))
print(f"stage-1: {b1.best_iteration} rounds, {len(f1)} features ({time.time()-t0:.0f}s)", flush=True)

# ---------------- group features on val
build(A / "models/lgb_s1.txt", A / "features/val_v2", A / "normalized", "train", A / "features/val_s2", log=lambda m: None)
print(f"val group features built ({time.time()-t0:.0f}s)", flush=True)
val = pl.read_parquet(A / "features/val_s2/*.parquet")
ids = split.filter(pl.col("role") == "val").select("s1_id", "country", "s1")
half = ids.with_columns((pl.col("s1").hash(seed=3) % 2).alias("h"))
A_ids, B_ids = half.filter(pl.col("h") == 0), half.filter(pl.col("h") == 1)
va, vb = val.join(A_ids.select("s1"), on="s1"), val.join(B_ids.select("s1"), on="s1")


def best_threshold(pred, sub):
    owned = one_owner(pred)
    res = [(t, evaluate(to_pairs(owned.filter(pl.col("p") >= t)), truth, sub.select("s1_id", "country")))
           for t in np.round(np.arange(0.5, 0.91, 0.05), 2)]
    t, r = max(res, key=lambda x: x[1]["f05"])
    return t, r


base = vb.select("s1", "m", "src").with_columns(pl.Series("p", vb["p1"].to_numpy()))
t, r = best_threshold(base, B_ids)
print(f"[B] stage-1 only        thr {t}: F0.5 {r['f05']:.4f} P {r['macro_p']:.4f} R {r['macro_r']:.4f} "
      f"US {r['by_country']['US']['f05']:.4f} IN {r['by_country']['India']['f05']:.4f}", flush=True)

# ---------------- stage 2 on half A -> eval half B
f2 = f1 + GROUP_COLS
easy = (pl.col("label") == 0) & (pl.col("p1") < 0.01)
tr = pl.concat([va.filter(~easy), va.filter(easy).sample(fraction=0.1, seed=1)])
w = np.where(tr.select(easy).to_series().to_numpy(), 10.0, 1.0)
b2 = lgb.train(PARAMS, lgb.Dataset(X(tr, f2), tr["label"].to_numpy(), weight=w, feature_name=f2), 800)
p2 = b2.predict(X(vb, f2))
t, r = best_threshold(vb.select("s1", "m", "src").with_columns(pl.Series("p", p2)), B_ids)
print(f"[B] stage-2 (group)     thr {t}: F0.5 {r['f05']:.4f} P {r['macro_p']:.4f} R {r['macro_r']:.4f} "
      f"US {r['by_country']['US']['f05']:.4f} IN {r['by_country']['India']['f05']:.4f}", flush=True)
imp = sorted(zip(f2, b2.feature_importance("gain")), key=lambda x: -x[1])
tot = sum(g for _, g in imp)
print("stage-2 gain top:", [(f, round(g / tot, 3)) for f, g in imp[:12]])
print(f"done ({time.time()-t0:.0f}s)")
