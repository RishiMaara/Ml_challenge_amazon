"""Stage 2 (v3): group-consistency features on top of stage-1 probabilities (Step 7.7).

An S1's true matches are near-copies of each other. For every candidate r of an S1 we look at the S1's best OTHER
candidates by stage-1 probability p1 (anchors a1, a2) and measure how similar r is to them and how confident they are:
    g_anchor1_p, g_anchor2_p, g_n_conf (#others with p1>=0.5), g_sum_p_others, g_p_rank, g_p_gap
    g_name_tset, g_name_key_eq, g_skel_tset, g_addr_tset, g_house_eq   (max over the two anchors)
Stage-2 LightGBM = stage-1 features + p1 + group features.

Leakage control: stage 2 is trained on fit S1 that stage 1 never saw (their p1 is out-of-sample).

Usage:
    python -m src.stage2 --stage1 ../../artifacts/models/lgb_v2.txt --features ../../artifacts/features/fit_sample2_v2 \
        --split train --out ../../artifacts/features/fit_sample2_v3
"""
import argparse
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist

from .features import load_records

GROUP_REC = ["name_core", "name_key", "name_skel", "addr_norm", "addr_house_no"]
GROUP_COLS = ["p1", "g_anchor1_p", "g_anchor2_p", "g_n_conf", "g_sum_p_others", "g_p_rank", "g_p_gap",
              "g_name_tset", "g_name_key_eq", "g_skel_tset", "g_addr_tset", "g_house_eq"]


def _sim(a, b, scorer):
    return cpdist(a, b, scorer=scorer, workers=-1, dtype=np.float32)


def group_features(part: pl.DataFrame, recs: pl.DataFrame) -> pl.DataFrame:
    """part: whole S1 groups with s1, m, src, p1. recs: m, src + GROUP_REC columns. Returns GROUP_COLS aligned to part."""
    part = part.with_row_index("_i")
    ranked = part.select("_i", "s1", "m", "src", "p1").sort(["s1", "p1"], descending=[False, True]).with_columns(
        pl.int_range(1, pl.len() + 1).over("s1").alias("r"))
    top = ranked.filter(pl.col("r") <= 3).select("s1", "r", pl.col("m").alias("am"), pl.col("src").alias("asrc"),
                                                  pl.col("p1").alias("ap"))
    # the two best OTHER candidates for each row
    anchors = (ranked.select("_i", "s1", "r").join(top, on="s1", suffix="_a")
                     .filter(pl.col("r") != pl.col("r_a"))
                     .sort(["_i", "r_a"]).with_columns(pl.int_range(1, pl.len() + 1).over("_i").alias("k"))
                     .filter(pl.col("k") <= 2))
    agg = part.group_by("s1").agg(pl.col("p1").sum().alias("_sum"), (pl.col("p1") >= 0.5).sum().alias("_nconf"))
    base = (ranked.select("_i", "s1", "r", "p1").join(agg, on="s1")
                  .join(anchors.filter(pl.col("k") == 1).select("_i", pl.col("ap").alias("g_anchor1_p")), on="_i", how="left")
                  .join(anchors.filter(pl.col("k") == 2).select("_i", pl.col("ap").alias("g_anchor2_p")), on="_i", how="left")
                  .with_columns(
                      (pl.col("_nconf") - (pl.col("p1") >= 0.5).cast(pl.UInt32)).alias("g_n_conf"),
                      (pl.col("_sum") - pl.col("p1")).alias("g_sum_p_others"),
                      pl.col("r").cast(pl.UInt16).alias("g_p_rank"),
                      (pl.col("g_anchor1_p") - pl.col("p1")).alias("g_p_gap")))
    # string similarity of the row's record vs each anchor's record
    rr = recs.rename({c: c + "_r" for c in GROUP_REC})
    ra = recs.rename({c: c + "_a" for c in GROUP_REC} | {"m": "am", "src": "asrc"})
    pairs = (anchors.select("_i", "k", "am", "asrc").join(part.select("_i", "m", "src"), on="_i")
                    .join(rr, on=["m", "src"], how="left").join(ra, on=["am", "asrc"], how="left"))
    sims = pairs.select("_i").with_columns(
        pl.Series("nt", _sim(pairs["name_core_r"].fill_null("").to_list(), pairs["name_core_a"].fill_null("").to_list(), fuzz.token_set_ratio)),
        pl.Series("st", _sim(pairs["name_skel_r"].fill_null("").to_list(), pairs["name_skel_a"].fill_null("").to_list(), fuzz.token_set_ratio)),
        pl.Series("at", _sim(pairs["addr_norm_r"].fill_null("").to_list(), pairs["addr_norm_a"].fill_null("").to_list(), fuzz.token_set_ratio)),
        (pairs["name_key_r"] == pairs["name_key_a"]).cast(pl.Int8).alias("ke"),
        pl.when((pairs["addr_house_no_r"] != "") & (pairs["addr_house_no_a"] != ""))
          .then((pairs["addr_house_no_r"] == pairs["addr_house_no_a"]).cast(pl.Int8)).otherwise(None).alias("he"),
    ).group_by("_i").agg(pl.col("nt").max().alias("g_name_tset"), pl.col("ke").max().alias("g_name_key_eq"),
                         pl.col("st").max().alias("g_skel_tset"), pl.col("at").max().alias("g_addr_tset"),
                         pl.col("he").max().alias("g_house_eq"))
    out = base.join(sims, on="_i", how="left").sort("_i")
    return out.select(GROUP_COLS)


def build(stage1: Path, feat_dir: Path, norm_dir: Path, split: str, out_dir: Path, log=print) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.parquet"):
        old.unlink()
    meta = json.loads(stage1.with_suffix(".json").read_text())
    feats = meta["features"]
    booster = lgb.Booster(model_file=str(stage1))
    t0 = time.time()
    parts = sorted(Path(feat_dir).glob("*.parquet"))
    for i, p in enumerate(parts):
        df = pl.read_parquet(p)
        X = df.select([pl.col(c).cast(pl.Float32) for c in feats]).to_numpy()
        df = df.with_columns(pl.Series("p1", booster.predict(X), dtype=pl.Float32))
        recs = pl.concat([load_records(norm_dir, split, s, df.filter(pl.col("src") == s)["m"].unique()) for s in (2, 3)])
        recs = recs.select(["m", "src"] + GROUP_REC)
        g = group_features(df.select("s1", "m", "src", "p1"), recs)
        pl.concat([df.drop("p1"), g], how="horizontal").write_parquet(out_dir / p.name)
        if i % 10 == 0:
            log(f"  part {i + 1}/{len(parts)} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1", type=Path, required=True)
    ap.add_argument("--features", type=Path, required=True)
    ap.add_argument("--norm", type=Path, default=Path("../../artifacts/normalized"))
    ap.add_argument("--split", choices=["train", "test"], required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    build(a.stage1, a.features, a.norm, a.split, a.out)
