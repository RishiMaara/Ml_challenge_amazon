"""v5 = v4b main pairs + blocking-extension pairs scored by a dedicated model (src/blockext.py).

    python notebooks/run_v5.py --main-tag v4b            # val: extension pairs, 2-fold model, combined OOF report
    python notebooks/run_v5.py --main-tag v4b --test     # + test extension pairs -> artifacts/test_scores_ext.parquet

Extension pairs have no blocking scores, so they get their own LightGBM: string similarities (src.features),
refiner features (src.refine), extension key stats, and context from the main model (is the record already
confidently owned by another S1? how many matches does this S1 already have?).
"""
import argparse
import gc
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code" / "business_entity_resolution"))
from src.blockext import extension_pairs, load  # noqa: E402
from src.evaluate import evaluate  # noqa: E402
from src.features import REC_COLS, load_records, pair_features  # noqa: E402
from src.refine import NEW_COLS, PARAMS3, pair_new_features  # noqa: E402
from src.train import one_owner, to_pairs  # noqa: E402

A = ROOT / "artifacts"
R = A / "refine"
t0 = time.time()
log = lambda m: print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)  # noqa: E731
ap = argparse.ArgumentParser()
ap.add_argument("--main-tag", default="v4b")
ap.add_argument("--test", action="store_true")
ap.add_argument("--top", type=int, default=3)
a = ap.parse_args()
num = lambda c: pl.col(c).str.slice(3).cast(pl.UInt32)  # noqa: E731


def gen_pairs(split, s1_ids, cand_glob, out):
    """One country at a time (the full test pool with list keys does not fit in 16 GB); per-country parts are cached."""
    if out.exists():
        return pl.read_parquet(out)
    countries = pl.read_parquet(A / "normalized" / f"{split}_source1.parquet", columns=["country"])["country"].unique().sort().to_list()
    parts = []
    for c in countries:
        part = out.with_name(f"{out.stem}_{c}.parquet")
        if not part.exists():
            s1c = load(A / "normalized", R, split, 1, c)
            if s1_ids is not None:
                s1c = s1c.filter(pl.col("s1").is_in(s1_ids.implode()))
            if s1c.height == 0:
                continue
            pool = pl.concat([load(A / "normalized", R, split, s, c) for s in (2, 3)])
            ex = pl.scan_parquet(cand_glob).select("s1", "m", "src").filter(pl.col("s1").is_in(s1c["s1"].implode())).unique().collect()
            extension_pairs(s1c, pool, ex, top=a.top).write_parquet(part)
            del ex, pool, s1c
            gc.collect()
        parts.append(pl.read_parquet(part))
        log(f"  {split} {c}: {parts[-1].height:,} extension pairs")
    e = pl.concat(parts)
    e.write_parquet(out)
    return e


def featurize(split, e, main, out):
    """main: (s1, m, src, p) main-model scores for the same split (context only)."""
    if out.exists():
        return pl.read_parquet(out)
    s1t, pool = pl.read_parquet(R / f"{split}_s1.parquet"), pl.read_parquet(R / f"{split}_pool.parquet")
    r1 = load_records(A / "normalized", split, 1, e["s1"].unique()).rename({c: c + "_1" for c in REC_COLS})
    r2 = pl.concat([load_records(A / "normalized", split, s, e.filter(pl.col("src") == s)["m"].unique()) for s in (2, 3)])
    r2 = r2.rename({c: c + "_2" for c in REC_COLS})
    parts = []
    for i in range(0, e.height, 1_000_000):
        chunk = e.slice(i, 1_000_000)
        wide = chunk.join(r1, on="s1", how="left").join(r2, on=["m", "src"], how="left")
        wide = wide.with_columns([pl.col(c + s).fill_null("") for c in REC_COLS if c not in ("name_is_domain", "name_indic", "addr_empty")
                                  for s in ("_1", "_2")])
        f = pl.concat([chunk, pair_features(wide)], how="horizontal")
        parts.append(pair_new_features(f, s1t, pool))
    d = pl.concat(parts)
    own = one_owner(main)
    rec = main.group_by("m", "src").agg(pl.col("p").max().alias("x_rec_pmax"), (pl.col("p") >= 0.5).sum().alias("x_rec_nconf"))
    s1c = own.group_by("s1").agg((pl.col("p") >= 0.75).sum().alias("x_s1_nsel"), pl.col("p").max().alias("x_s1_pmax"))
    d = (d.join(rec, on=["m", "src"], how="left").join(s1c, on="s1", how="left")
          .with_columns(pl.col("x_rec_pmax", "x_s1_pmax").fill_null(0.0), pl.col("x_rec_nconf", "x_s1_nsel").fill_null(0)))
    d = d.with_columns(pl.col(pl.Float64).cast(pl.Float32))
    d.write_parquet(out)
    return d


ID = {"s1", "m", "src", "y", "label"}
split = pl.read_parquet(A / "splits/split_v1.parquet")
val_ids = split.filter(pl.col("role") == "val")["s1_id"].str.slice(3).cast(pl.UInt32)
ev = gen_pairs("train", val_ids, str(A / "blocking/union_train/*.parquet"), R / "ext_val_pairs.parquet")
log(f"val extension pairs {ev.height:,}")
main_v = (pl.read_parquet(R / f"val_oof_{a.main_tag}.parquet", columns=["s1", "m", "src", "p"]))
dv = featurize("train", ev, main_v, R / "ext_val_set.parquet")
truth = pl.read_parquet(A / "processed/train_gt_pairs.parquet")
tp = truth.drop_nulls("match_id").select(num("s1_id").alias("s1"), num("match_id").alias("m"),
                                         pl.col("match_id").str.slice(1, 1).cast(pl.UInt8).alias("src")).with_columns(pl.lit(1, pl.Int8).alias("y"))
dv = dv.drop("y", strict=False).join(tp, on=["s1", "m", "src"], how="left").with_columns(pl.col("y").fill_null(0))
feats = [c for c in dv.columns if c not in ID and dv[c].dtype != pl.Utf8]
log(f"val ext set {dv.height:,} pairs, pos {dv['y'].sum():,}, {len(feats)} features")
X = dv.select([pl.col(c).cast(pl.Float32) for c in feats]).to_numpy()
y = dv["y"].to_numpy()
fold = (dv["s1"].hash(seed=3) % 2).to_numpy()
inner = (dv["s1"].hash(seed=9) % 10).to_numpy() == 0
oof = np.zeros(dv.height, dtype=np.float32)
models = []
for k in (0, 1):
    fit, es = (fold != k) & ~inner, (fold != k) & inner
    b = lgb.train(PARAMS3, lgb.Dataset(X[fit], y[fit], feature_name=feats), 3000,
                  valid_sets=[lgb.Dataset(X[es], y[es])], callbacks=[lgb.early_stopping(150, verbose=False)])
    oof[fold == k] = b.predict(X[fold == k], num_iteration=b.best_iteration)
    b.save_model(str(A / f"models/lgb_ext_f{k}.txt"), num_iteration=b.best_iteration)
    models.append(b)
    log(f"ext fold {k}: {b.best_iteration} rounds")
imp = sorted(zip(feats, models[0].feature_importance("gain")), key=lambda x: -x[1])
log("ext top gain: " + ", ".join(f"{f} {g/sum(x[1] for x in imp):.3f}" for f, g in imp[:15]))
ext_v = dv.select("s1", "m", "src").with_columns(pl.Series("p", oof))
ext_v.write_parquet(R / "ext_val_oof.parquet")

ids = split.filter(pl.col("role") == "val").select("s1_id", "country")
res = {}
for mt in (0.7, 0.75):
    base = evaluate(to_pairs(one_owner(main_v).filter(pl.col("p") >= mt)), truth, ids)
    log(f"main {a.main_tag} @{mt}: F0.5 {base['f05']:.4f} US {base['by_country']['US']['f05']:.4f} IN {base['by_country']['India']['f05']:.4f}")
    for et in (0.5, 0.6, 0.7, 0.8, 0.9):
        comb = pl.concat([main_v.with_columns(pl.lit(mt).alias("thr")), ext_v.with_columns(pl.lit(et).alias("thr"))])
        r = evaluate(to_pairs(one_owner(comb).filter(pl.col("p") >= pl.col("thr"))), truth, ids)
        res[(mt, et)] = r["f05"]
        log(f"  + ext @{et}: F0.5 {r['f05']:.4f} P {r['macro_p']:.4f} R {r['macro_r']:.4f} "
            f"US {r['by_country']['US']['f05']:.4f} IN {r['by_country']['India']['f05']:.4f}")
best = max(res, key=res.get)
(A / "models/lgb_ext.json").write_text(json.dumps({"features": feats, "main_thr": best[0], "ext_thr": best[1], "val_f05": res[best]}))
log(f"best main/ext thr {best}: {res[best]:.4f}")

if a.test:
    et_ = gen_pairs("test", None, str(A / "blocking/union_test/*.parquet"), R / "ext_test_pairs.parquet")
    log(f"test extension pairs {et_.height:,}")
    main_t = pl.read_parquet(A / f"test_scores_{a.main_tag}.parquet", columns=["s1", "m", "src", "p"])
    dt = featurize("test", et_, main_t, R / "ext_test_set.parquet")
    Xt = dt.select([pl.col(c).cast(pl.Float32) for c in feats]).to_numpy()
    pt = np.mean([m.predict(Xt) for m in models], axis=0)
    dt.select("s1", "m", "src").with_columns(pl.Series("p", pt, dtype=pl.Float32)).write_parquet(A / "test_scores_ext.parquet")
    log("test extension scored")
