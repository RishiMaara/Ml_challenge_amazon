"""Validation splits over TRAIN Source-1 entities.

Blocking/features run over ALL train S1 entities (label-free, so no leakage and realistic competition
between S1s for each S2/S3 record). The split only decides whose LABELS train the model vs. score it:
  * fold 0..4  : stratified by (country, match-count bucket); fold 0 = holdout "val", 1..4 = "fit"
  * dev        : a fixed 20k-entity subsample of val for fast iteration
  * LOCO       : leave-one-country-out (fit on one country's labels, score the other) - France proxy

Usage:
    python -m src.split --processed ../../artifacts/processed --out ../../artifacts/splits
"""
import argparse
from pathlib import Path

import polars as pl

N_FOLDS = 5
SEED = 2026
DEV_SIZE = 20_000


def make_split(processed: Path) -> pl.DataFrame:
    s1 = pl.read_parquet(processed / "train_source1.parquet").select(pl.col("entity_id").alias("s1_id"), "country")
    gt = pl.read_parquet(processed / "train_gt_pairs.parquet")
    n = gt.group_by("s1_id").agg(pl.col("match_id").drop_nulls().len().alias("n_true"))
    df = s1.join(n, on="s1_id", how="left").with_columns(pl.col("n_true").fill_null(0))
    df = df.with_columns(pl.col("n_true").clip(0, 6).alias("bucket"))
    # stratified fold assignment: shuffle within stratum, then round-robin
    df = (df.with_columns(pl.int_range(pl.len()).shuffle(seed=SEED).over("country", "bucket").alias("r"))
            .with_columns((pl.col("r") % N_FOLDS).cast(pl.Int8).alias("fold"))
            .drop("r", "bucket"))
    df = df.with_columns(pl.when(pl.col("fold") == 0).then(pl.lit("val")).otherwise(pl.lit("fit")).alias("role"))
    dev_ids = df.filter(pl.col("role") == "val").sample(DEV_SIZE, seed=SEED)["s1_id"].implode()
    return df.with_columns(pl.col("s1_id").is_in(dev_ids).alias("is_dev")).sort("s1_id")


def subsets(split: pl.DataFrame) -> dict[str, tuple[pl.DataFrame, pl.DataFrame]]:
    """Named (fit_ids, eval_ids) pairs. LOCO fits on one country's fit-role labels and scores the other's val."""
    fit, val = split.filter(pl.col("role") == "fit"), split.filter(pl.col("role") == "val")
    out = {"val": (fit, val), "dev": (fit, val.filter(pl.col("is_dev")))}
    for c in split["country"].unique().to_list():
        out[f"loco_{c}"] = (fit.filter(pl.col("country") != c), val.filter(pl.col("country") == c))
    return {k: (f.select("s1_id", "country"), e.select("s1_id", "country")) for k, (f, e) in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    df = make_split(args.processed)
    df.write_parquet(args.out / "split_v1.parquet")
    print(df.group_by("role", "country").agg(pl.len(), (pl.col("n_true") == 0).mean().alias("singleton_rate"),
                                              pl.col("n_true").mean().alias("mean_true"), pl.col("is_dev").sum().alias("dev"))
            .sort("role", "country"))


if __name__ == "__main__":
    main()
