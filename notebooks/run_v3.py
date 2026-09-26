"""v3 pipeline: stage-1 (no crowd features) -> group features -> stage-2 (600k fresh fit S1) -> val eval -> test.

    python notebooks/run_v3.py --tag v2            # blocking candidates only (features/*_v2)
    python notebooks/run_v3.py --tag u  --retrain-s1   # union candidates incl. Kaggle neighbours (features/*_u)

Inputs  features/{fit_sample,fit_sample2,dev,val,test}_{tag}
Outputs models/lgb_s1_{tag}, models/lgb_v3_{tag}, features/*_s2{tag}, a val report, and (with --test)
        output/matching_results.tsv via src.predict (unseen-country safeguard on).
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code" / "business_entity_resolution"
sys.path.insert(0, str(CODE))
from src.evaluate import evaluate  # noqa: E402
from src.features import CROWD_COLS  # noqa: E402
from src.stage2 import GROUP_COLS, build  # noqa: E402
from src.train import PARAMS, feature_names, one_owner, to_pairs  # noqa: E402

A = ROOT / "artifacts"
F, M = A / "features", A / "models"
X = lambda d, f: d.select([pl.col(c).cast(pl.Float32) for c in f]).to_numpy()  # noqa: E731
t0 = time.time()
log = lambda m: print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)  # noqa: E731

ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="v2")
ap.add_argument("--retrain-s1", action="store_true")
ap.add_argument("--test", action="store_true", help="also build test group features and predict")
ap.add_argument("--easy-keep", type=float, default=0.1)
a = ap.parse_args()
tag = a.tag

# ---------------- stage 1
s1_path = M / f"lgb_s1_{tag}.txt" if a.retrain_s1 else M / "lgb_s1.txt"
if a.retrain_s1:
    fit, dev = pl.read_parquet(F / f"fit_sample_{tag}/*.parquet"), pl.read_parquet(F / f"dev_{tag}/*.parquet")
    f1 = [c for c in feature_names(fit) if c not in CROWD_COLS]
    b1 = lgb.train(PARAMS, lgb.Dataset(X(fit, f1), fit["label"].to_numpy(), feature_name=f1), 3000,
                   valid_sets=[lgb.Dataset(X(dev, f1), dev["label"].to_numpy())], callbacks=[lgb.early_stopping(100, verbose=False)])
    del fit, dev
    b1.save_model(str(s1_path))
    s1_path.with_suffix(".json").write_text(json.dumps({"features": f1, "best_iteration": b1.best_iteration}))
    log(f"stage-1 retrained: {b1.best_iteration} rounds, {len(f1)} features")

# ---------------- group features (skip sets already built)
for name in ("fit_sample2", "dev", "val"):
    out = F / f"{name}_s2{tag}"
    if not any(out.glob("*.parquet")):
        build(s1_path, F / f"{name}_{tag}", A / "normalized", "train", out, log=lambda m: None)
    log(f"group features ready: {out.name}")

# ---------------- stage 2
f1 = json.loads(s1_path.with_suffix(".json").read_text())["features"]
f2 = f1 + GROUP_COLS
easy = (pl.col("label") == 0) & (pl.col("p1") < 0.01)
chunks, n_all = [], 0
for i, p in enumerate(sorted((F / f"fit_sample2_s2{tag}").glob("*.parquet"))):  # down-sample easy negatives per part
    d = pl.read_parquet(p, columns=f2 + ["label"])
    n_all += d.height
    chunks.append(pl.concat([d.filter(~easy), d.filter(easy).sample(fraction=a.easy_keep, seed=i)])
                    .with_columns(pl.col(pl.Float64).cast(pl.Float32)))
tr = pl.concat(chunks)
del chunks
w = np.where(tr.select(easy).to_series().to_numpy(), 1.0 / a.easy_keep, 1.0)
log(f"stage-2 train rows {tr.height:,} (from {n_all:,}; pos {tr['label'].sum():,})")
dev = pl.read_parquet(F / f"dev_s2{tag}/*.parquet")
b2 = lgb.train(PARAMS, lgb.Dataset(X(tr, f2), tr["label"].to_numpy(), weight=w, feature_name=f2), 3000,
               valid_sets=[lgb.Dataset(X(dev, f2), dev["label"].to_numpy())],
               callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(200)])
del tr, dev
v3_path = M / f"lgb_v3_{tag}.txt"
b2.save_model(str(v3_path))
log(f"stage-2 trained: {b2.best_iteration} rounds")

# ---------------- full val
split = pl.read_parquet(A / "splits/split_v1.parquet")
truth = pl.read_parquet(A / "processed/train_gt_pairs.parquet")
ids = split.filter(pl.col("role") == "val").select("s1_id", "country")
parts = []
for p in sorted((F / f"val_s2{tag}").glob("*.parquet")):
    d = pl.read_parquet(p)
    parts.append(d.select("s1", "m", "src").with_columns(pl.Series("p", b2.predict(X(d, f2)), dtype=pl.Float32),
                                                         pl.Series("p1", d["p1"].to_numpy())).filter(pl.col("p") >= 0.01))
pv = pl.concat(parts)
res = []
for t in np.round(np.arange(0.5, 0.91, 0.05), 2):
    r = evaluate(to_pairs(one_owner(pv).filter(pl.col("p") >= t)), truth, ids)
    res.append((float(t), r))
    log(f"val thr {t}: F0.5 {r['f05']:.4f} P {r['macro_p']:.4f} R {r['macro_r']:.4f} "
        f"US {r['by_country']['US']['f05']:.4f} IN {r['by_country']['India']['f05']:.4f}")
thr, best = max(res, key=lambda x: x[1]["f05"])
s1only = max(evaluate(to_pairs(one_owner(pv.with_columns(pl.col("p1").alias("p"))).filter(pl.col("p") >= t)), truth, ids)["f05"]
             for t in (0.65, 0.7, 0.75, 0.8))
log(f"BEST v3 val F0.5 {best['f05']:.4f} @thr {thr} (stage-1 alone {s1only:.4f})")
v3_path.with_suffix(".json").write_text(json.dumps({"features": f2, "best_iteration": b2.best_iteration,
                                                     "threshold": thr, "val_f05": best["f05"], "stage1": str(s1_path)}))

# ---------------- test
if a.test:
    out = F / f"test_s2{tag}"
    build(s1_path, F / f"test_{tag}", A / "normalized", "test", out, log=log)
    log("test group features built")
    subprocess.run([sys.executable, "-m", "src.predict", "--features", str(out), "--model", str(v3_path),
                    "--out", str(ROOT / "output"), "--scores-out", str(A / f"test_scores_v3_{tag}.parquet")],
                   cwd=CODE, check=True)
    log("test predictions written")
