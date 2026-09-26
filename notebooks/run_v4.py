"""v4 = v3 + stage-3 sibling-aware refiner (src/refine.py).

    python notebooks/run_v4.py            # build val set, 2-fold train, OOF val report
    python notebooks/run_v4.py --test     # + score test pairs, write output/matching_results.tsv

Inputs : artifacts/features/{val,test}_s2u (v3 feature rows), models/lgb_v3_u, artifacts/{val_scores_v3u,test_scores_v3_u}.parquet
Outputs: artifacts/refine/*, models/lgb_v4_f{0,1}.txt, val report, test submission.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code" / "business_entity_resolution"
sys.path.insert(0, str(CODE))
from src.evaluate import evaluate, write_id_list_tsv  # noqa: E402
from src.refine import NEW_COLS, PARAMS3, pair_new_features, support_tables  # noqa: E402
from src.train import one_owner, to_pairs  # noqa: E402

A = ROOT / "artifacts"
R = A / "refine"
R.mkdir(exist_ok=True)
FLOOR = 0.02
t0 = time.time()
log = lambda m: print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)  # noqa: E731

ap = argparse.ArgumentParser()
ap.add_argument("--test", action="store_true")
ap.add_argument("--rebuild", action="store_true")
ap.add_argument("--drop", nargs="*", default=[])
ap.add_argument("--tag", default="v4")
a = ap.parse_args()

v3_feats = json.loads((A / "models/lgb_v3_u.json").read_text())["features"]


def tables(split):
    f1, f2 = R / f"{split}_s1.parquet", R / f"{split}_pool.parquet"
    if not f1.exists() or a.rebuild:
        s1, pool = support_tables(A / "normalized", A / "processed", split)
        s1.write_parquet(f1)
        pool.write_parquet(f2)
        log(f"{split} support tables built")
    return pl.read_parquet(f1), pl.read_parquet(f2)


def build_set(split, feat_dir, scores_path, out):
    if out.exists() and not a.rebuild:
        return pl.read_parquet(out)
    s1, pool = tables(split)
    sc = pl.read_parquet(scores_path).filter(pl.col("p") >= FLOOR).select("s1", "m", "src", pl.col("p").alias("p3"))
    cols = ["s1", "m", "src"] + [c for c in v3_feats if c != "src"] + (["label"] if split == "train" else [])
    parts = []
    for p in sorted(Path(feat_dir).glob("*.parquet")):
        d = pl.read_parquet(p, columns=cols).join(sc, on=["s1", "m", "src"], how="inner")
        parts.append(d.with_columns(pl.col(pl.Float64).cast(pl.Float32)))
    d = pl.concat(parts)
    log(f"{split}: {d.height:,} pairs with p3>={FLOOR}; adding new features")
    d = pair_new_features(d, s1, pool)
    d.write_parquet(out)
    log(f"{split} set written")
    return d


val = build_set("train", A / "features/val_s2u", A / "val_scores_v3u.parquet", R / "val_set.parquet")
feats = [c for c in v3_feats + ["p3"] + NEW_COLS if c not in a.drop]
X = lambda d: d.select([pl.col(c).cast(pl.Float32) for c in feats]).to_numpy()  # noqa: E731
fold = (val["s1"].hash(seed=3) % 2).to_numpy()
y = val["label"].to_numpy()
oof = np.zeros(val.height, dtype=np.float32)
models = []
for k in (0, 1):
    tr, te = fold != k, fold == k
    # inner early-stopping split inside the training fold
    inner = (val["s1"].hash(seed=9) % 10).to_numpy() == 0
    fit, es = tr & ~inner, tr & inner
    Xa = X(val)
    b = lgb.train(PARAMS3, lgb.Dataset(Xa[fit], y[fit], feature_name=feats), 4000,
                  valid_sets=[lgb.Dataset(Xa[es], y[es])], callbacks=[lgb.early_stopping(150, verbose=False)])
    oof[te] = b.predict(Xa[te], num_iteration=b.best_iteration)
    b.save_model(str(A / f"models/lgb_{a.tag}_f{k}.txt"), num_iteration=b.best_iteration)
    models.append(b)
    log(f"fold {k}: {b.best_iteration} rounds")
    del Xa
imp = sorted(zip(feats, models[0].feature_importance("gain")), key=lambda x: -x[1])
tot = sum(g for _, g in imp)
log("top gain: " + ", ".join(f"{f} {g/tot:.3f}" for f, g in imp[:25]))

split = pl.read_parquet(A / "splits/split_v1.parquet")
ids = split.filter(pl.col("role") == "val").select("s1_id", "country")
truth = pl.read_parquet(A / "processed/train_gt_pairs.parquet")
pv = val.select("s1", "m", "src", "p3").with_columns(pl.Series("p", oof))
base = max((evaluate(to_pairs(one_owner(pv, "p3").filter(pl.col("p3") >= t)), truth, ids)["f05"], t) for t in (0.6, 0.7, 0.8))
log(f"v3 baseline on the same pairs: F0.5 {base[0]:.4f} @ {base[1]}")
res = []
for t in np.round(np.arange(0.4, 0.96, 0.05), 2):
    r = evaluate(to_pairs(one_owner(pv).filter(pl.col("p") >= t)), truth, ids)
    res.append((float(t), r["f05"]))
    log(f"v4 thr {t}: F0.5 {r['f05']:.4f} P {r['macro_p']:.4f} R {r['macro_r']:.4f} "
        f"US {r['by_country']['US']['f05']:.4f} IN {r['by_country']['India']['f05']:.4f}")
thr = max(res, key=lambda x: x[1])[0]
(A / f"models/lgb_{a.tag}.json").write_text(json.dumps({"features": feats, "threshold": thr, "val_f05": max(r for _, r in res)}))
pv.write_parquet(R / f"val_oof_{a.tag}.parquet")

if a.test:
    test = build_set("test", A / "features/test_s2u", A / "test_scores_v3_u.parquet", R / "test_set.parquet")
    Xt = X(test)
    pt = np.mean([m.predict(Xt) for m in models], axis=0)
    pred = test.select("s1", "m", "src", "p3", "hv_diff", "hv_eq").with_columns(pl.Series("p", pt, dtype=pl.Float32))
    pred.write_parquet(A / f"test_scores_{a.tag}.parquet")
    log("test scored")
