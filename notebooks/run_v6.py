"""v6 = v5 + name-only extension for unclaimed empty-address records.

Empty-address S2/S3 records are 97.7% true matches in training (distractors almost always carry an address), but
~45k of them on test are claimed by no S1 (the main blocker ranks them low: the name is the only evidence and it
often carries a typo). For every such record we look up S1 of the same country sharing a rare name word / skeleton,
score name similarity, and let a dedicated LightGBM decide, using how clearly the best S1 beats the runner-up.

    python notebooks/run_v6.py            # val: 2-fold OOF + combined report (main v4c + ext + v6)
    python notebooks/run_v6.py --test     # + test -> artifacts/test_scores_name.parquet
"""
import argparse
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler
from rapidfuzz.process import cpdist

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code" / "business_entity_resolution"))
from src.evaluate import evaluate  # noqa: E402
from src.refine import PARAMS3, _acronym, _name_diff, token_df  # noqa: E402
from src.train import one_owner, to_pairs  # noqa: E402

A = ROOT / "artifacts"
R = A / "refine"
N = A / "normalized"
t0 = time.time()
log = lambda m: print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)  # noqa: E731
ap = argparse.ArgumentParser()
ap.add_argument("--test", action="store_true")
ap.add_argument("--top", type=int, default=10)
ap.add_argument("--claim-max", type=float, default=0.3, help="records whose best main/ext p is below this are targets")
a = ap.parse_args()
num = lambda c: pl.col(c).str.slice(3).cast(pl.UInt32)  # noqa: E731
COLS = ["entity_id", "country", "name_core", "name_skel", "name_key", "name_norm", "name_legal"]


def toks() -> pl.Expr:
    w = pl.col("name_core").str.split(" ").list.eval(pl.element().filter(pl.element().str.len_chars() >= 3)).list.eval(pl.lit("w|") + pl.element())
    k = pl.col("name_skel").str.split(" ").list.eval(pl.element().filter(pl.element().str.len_chars() >= 3)).list.eval(pl.lit("k|") + pl.element())
    return pl.concat_list(w, k).list.unique().alias("tk")


def s1_table(split):
    s = pl.read_parquet(N / f"{split}_source1.parquet", columns=COLS).with_columns(num("entity_id").alias("s1")).drop("entity_id")
    kc = s.group_by("country", "name_key").len().rename({"len": "s1_n_key"})
    return s.join(kc, on=["country", "name_key"]).with_columns(toks())


def rec_table(split):
    r = pl.concat([pl.read_parquet(N / f"{split}_source{i}.parquet", columns=COLS + ["addr_empty"])
                     .filter(pl.col("addr_empty")).with_columns(num("entity_id").alias("m"), pl.lit(i, pl.UInt8).alias("src"))
                   for i in (2, 3)]).drop("entity_id", "addr_empty")
    return r.with_columns(toks())


def gen(s1, rec, max_df=2000):
    out = []
    for c in rec["country"].unique().to_list():
        st = s1.filter(pl.col("country") == c).select("s1", "tk").explode("tk").drop_nulls()
        df = st.group_by("tk").len()
        st = st.join(df.filter(pl.col("len") <= max_df), on="tk").with_columns((1.0 / (1.0 + pl.col("len").log())).alias("w"))
        rt = rec.filter(pl.col("country") == c).select("m", "src", "tk").explode("tk").drop_nulls()
        pr = (rt.join(st.select("s1", "tk", "w"), on="tk").group_by("m", "src", "s1").agg(pl.col("w").sum().alias("tw"), pl.len().alias("tn"))
                .sort("tw", descending=True).group_by("m", "src").head(a.top))
        out.append(pr)
        log(f"  {c}: {pr.height:,} name pairs for {rt.select('m', 'src').unique().height:,} records")
    return pl.concat(out)


def featurize(pairs, s1, rec, main_sel):
    w = (pairs.join(s1.drop("tk").rename({c: c + "_1" for c in COLS[2:]}), on="s1")
              .join(rec.drop("tk", "country").rename({c: c + "_2" for c in COLS[2:]}), on=["m", "src"]))
    n1, n2 = w["name_core_1"].to_list(), w["name_core_2"].to_list()
    c1, c2 = [x.replace(" ", "") for x in n1], [x.replace(" ", "") for x in n2]
    f = {"q_tset": cpdist(n1, n2, scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float32),
         "q_tsort": cpdist(n1, n2, scorer=fuzz.token_sort_ratio, workers=-1, dtype=np.float32),
         "q_ratio": cpdist(n1, n2, scorer=fuzz.ratio, workers=-1, dtype=np.float32),
         "q_jw": cpdist(n1, n2, scorer=JaroWinkler.normalized_similarity, workers=-1, dtype=np.float32),
         "q_cat": cpdist(c1, c2, scorer=fuzz.ratio, workers=-1, dtype=np.float32),
         "q_skel": cpdist(w["name_skel_1"].to_list(), w["name_skel_2"].to_list(), scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float32),
         "q_norm": cpdist(w["name_norm_1"].to_list(), w["name_norm_2"].to_list(), scorer=fuzz.token_sort_ratio, workers=-1, dtype=np.float32)}
    dfs = token_df(s1.select("country", "name_core"))
    nd = np.array([_name_diff(x, y, "", dfs.get(c, {})) for x, y, c in zip(n1, n2, w["country"].to_list())], dtype=np.float32)
    acr = [_acronym(x, y, z) for x, y, z in zip(w["name_norm_1"].to_list(), n1, w["name_norm_2"].to_list())]
    w = w.with_columns(**{k: pl.Series(v) for k, v in f.items()},
                       **{k: pl.Series(nd[:, i]) for i, k in enumerate(["q_miss1", "q_extra2", "q_len1", "q_len2", "q_miss1_df", "q_extra2_df"])},
                       q_acr=pl.Series(acr, dtype=pl.Int8),
                       q_key_eq=(pl.col("name_key_1") == pl.col("name_key_2")).cast(pl.Int8),
                       q_legal_eq=pl.when((pl.col("name_legal_1") != "") & (pl.col("name_legal_2") != ""))
                                    .then((pl.col("name_legal_1") == pl.col("name_legal_2")).cast(pl.Int8)))
    w = w.with_columns((pl.col("q_tset") + pl.col("q_cat") + pl.col("q_skel")).alias("q_sum"))
    g = ["m", "src"]
    w = w.with_columns(
        pl.col("q_sum").rank("min", descending=True).over(g).cast(pl.Int16).alias("q_rank"),
        (pl.col("q_sum") - pl.col("q_sum").max().over(g)).alias("q_gap_best"),
        pl.len().over(g).cast(pl.Int16).alias("q_ncand"),
        (pl.col("q_sum") >= pl.col("q_sum").max().over(g) - 5).sum().over(g).cast(pl.Int16).alias("q_n_near"),
        pl.col("q_key_eq").sum().over(g).cast(pl.Int16).alias("q_n_keyeq"),
    )
    w = w.with_columns(  # margin over the runner-up (for the best candidate) / to the best (others)
        pl.when(pl.col("q_rank") == 1).then(pl.col("q_sum") - pl.col("q_sum").sort(descending=True).slice(1, 1).first().over(g))
          .otherwise(pl.col("q_gap_best")).fill_null(300.0).alias("q_margin"))
    sn = main_sel.group_by("s1").agg(pl.len().alias("q_s1_nsel"))
    w = w.join(sn, on="s1", how="left").with_columns(pl.col("q_s1_nsel").fill_null(0))
    return w


FEATS = ["tw", "tn", "q_tset", "q_tsort", "q_ratio", "q_jw", "q_cat", "q_skel", "q_norm", "q_miss1", "q_extra2", "q_len1", "q_len2",
         "q_miss1_df", "q_extra2_df", "q_acr", "q_key_eq", "q_legal_eq", "q_sum", "q_rank", "q_gap_best", "q_ncand", "q_n_near",
         "q_n_keyeq", "q_margin", "q_s1_nsel", "s1_n_key"]


def targets(rec, scored, claim_max):
    best = scored.group_by("m", "src").agg(pl.col("p").max().alias("pmax"))
    return rec.join(best, on=["m", "src"], how="left").filter(pl.col("pmax").fill_null(0) < claim_max).drop("pmax")


# ------------------------------------------------------------------ validation
split = pl.read_parquet(A / "splits/split_v1.parquet")
val_ids = split.filter(pl.col("role") == "val")["s1_id"].str.slice(3).cast(pl.UInt32)
truth = pl.read_parquet(A / "processed/train_gt_pairs.parquet")
tp = truth.drop_nulls("match_id").select(num("s1_id").alias("s1"), num("match_id").alias("m"),
                                         pl.col("match_id").str.slice(1, 1).cast(pl.UInt8).alias("src"))
main_v = pl.read_parquet(R / "val_oof_v4c.parquet", columns=["s1", "m", "src", "p"])
ext_v = pl.read_parquet(R / "ext_val_oof.parquet")
out_v = R / "name_val_set.parquet"
if not out_v.exists():
    s1 = s1_table("train")
    rec = rec_table("train")
    # records owned by a fit S1 are assumed claimed by the main model (no main scores exist for fit S1)
    fit_owned = tp.filter(~pl.col("s1").is_in(val_ids.implode())).select("m", "src")
    rec = targets(rec.join(fit_owned, on=["m", "src"], how="anti"), pl.concat([main_v, ext_v]), a.claim_max)
    log(f"val target records: {rec.height:,}")
    pairs = gen(s1, rec)
    main_sel = one_owner(pl.concat([main_v, ext_v])).filter(pl.col("p") >= 0.75)
    dv = featurize(pairs, s1, rec, main_sel)
    dv = dv.join(tp.with_columns(pl.lit(1, pl.Int8).alias("y")), on=["s1", "m", "src"], how="left").with_columns(pl.col("y").fill_null(0))
    dv.select(["s1", "m", "src", "country", "y"] + FEATS).write_parquet(out_v)
dv = pl.read_parquet(out_v)
log(f"val name set {dv.height:,} pairs, pos {dv['y'].sum():,}")
X = dv.select([pl.col(c).cast(pl.Float32) for c in FEATS]).to_numpy()
y = dv["y"].to_numpy()
fold = (dv["m"].hash(seed=5) % 2).to_numpy()
inner = (dv["m"].hash(seed=11) % 10).to_numpy() == 0
oof = np.zeros(dv.height, dtype=np.float32)
models = []
for k in (0, 1):
    fit, es = (fold != k) & ~inner, (fold != k) & inner
    b = lgb.train(PARAMS3, lgb.Dataset(X[fit], y[fit], feature_name=FEATS), 3000,
                  valid_sets=[lgb.Dataset(X[es], y[es])], callbacks=[lgb.early_stopping(150, verbose=False)])
    oof[fold == k] = b.predict(X[fold == k], num_iteration=b.best_iteration)
    b.save_model(str(A / f"models/lgb_name_f{k}.txt"), num_iteration=b.best_iteration)
    models.append(b)
    log(f"name fold {k}: {b.best_iteration} rounds")
imp = sorted(zip(FEATS, models[0].feature_importance("gain")), key=lambda x: -x[1])
log("name top gain: " + ", ".join(f"{f} {g/sum(x[1] for x in imp):.3f}" for f, g in imp[:12]))
name_v = dv.select("s1", "m", "src").with_columns(pl.Series("p", oof)).filter(pl.col("s1").is_in(val_ids.implode()))
name_v.write_parquet(R / "name_val_oof.parquet")
ids = split.filter(pl.col("role") == "val").select("s1_id", "country")
base = pl.concat([main_v.with_columns(pl.lit(0.75).alias("thr")), ext_v.with_columns(pl.lit(0.6).alias("thr"))])
r = evaluate(to_pairs(one_owner(base).filter(pl.col("p") >= pl.col("thr"))), truth, ids)
log(f"v5 (main+ext): F0.5 {r['f05']:.4f} P {r['macro_p']:.4f} R {r['macro_r']:.4f} US {r['by_country']['US']['f05']:.4f} IN {r['by_country']['India']['f05']:.4f}")
res = {}
for nt in (0.5, 0.6, 0.7, 0.8, 0.9):
    comb = pl.concat([base, name_v.join(base, on=["s1", "m", "src"], how="anti").with_columns(pl.lit(nt).alias("thr"))])
    r = evaluate(to_pairs(one_owner(comb).filter(pl.col("p") >= pl.col("thr"))), truth, ids)
    res[nt] = r["f05"]
    log(f"  + name @{nt}: F0.5 {r['f05']:.4f} P {r['macro_p']:.4f} R {r['macro_r']:.4f} US {r['by_country']['US']['f05']:.4f} IN {r['by_country']['India']['f05']:.4f}")
nt = max(res, key=res.get)
(A / "models/lgb_name.json").write_text(json.dumps({"features": FEATS, "thr": nt, "val_f05": res[nt]}))

if a.test:
    main_t = pl.read_parquet(A / "test_scores_v4c.parquet", columns=["s1", "m", "src", "p"])
    ext_t = pl.read_parquet(A / "test_scores_ext.parquet", columns=["s1", "m", "src", "p"])
    s1 = s1_table("test")
    rec = targets(rec_table("test"), pl.concat([main_t, ext_t]), a.claim_max)
    log(f"test target records: {rec.height:,}")
    pairs = gen(s1, rec)
    main_sel = one_owner(pl.concat([main_t, ext_t])).filter(pl.col("p") >= 0.75)
    dt = featurize(pairs, s1, rec, main_sel)
    Xt = dt.select([pl.col(c).cast(pl.Float32) for c in FEATS]).to_numpy()
    pt = np.mean([m.predict(Xt) for m in models], axis=0)
    dt.select("s1", "m", "src").with_columns(pl.Series("p", pt, dtype=pl.Float32)).write_parquet(A / "test_scores_name.parquet")
    log("test name pairs scored")
