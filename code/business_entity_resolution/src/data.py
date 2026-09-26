"""Data loading: raw TSV -> parquet, plus loaders used by every later stage.

Usage:
    python -m src.data --raw-dir ../../student_resource/dataset --out-dir ../../artifacts/processed
"""
import argparse
from pathlib import Path

import polars as pl

SOURCES = ("source1", "source2", "source3")
SPLITS = ("train", "test")
SOURCE_SCHEMA = {
    "entity_id": pl.Utf8,
    "business_name": pl.Utf8,
    "business_address": pl.Utf8,
    "country": pl.Utf8,
}


def read_source_tsv(path: Path) -> pl.DataFrame:
    """Read one source TSV exactly as-is: tab-separated, no quoting, all strings, '' kept as ''."""
    df = pl.read_csv(
        path,
        separator="\t",
        quote_char=None,
        schema=SOURCE_SCHEMA,
        encoding="utf8",
    )
    return df.with_columns(pl.col(c).fill_null("") for c in SOURCE_SCHEMA)


def read_ground_truth_tsv(path: Path) -> pl.DataFrame:
    """Ground truth exploded to one row per (S1, matched id) pair; singletons keep a null match_id."""
    gt = pl.read_csv(
        path,
        separator="\t",
        quote_char=None,
        schema={"source1_entity_id": pl.Utf8, "matched_entity_ids": pl.Utf8},
    ).with_columns(pl.col("matched_entity_ids").fill_null(""))
    return (
        gt.with_columns(
            pl.when(pl.col("matched_entity_ids") == "")
            .then(pl.lit([], dtype=pl.List(pl.Utf8)))
            .otherwise(pl.col("matched_entity_ids").str.split(","))
            .alias("match_id")
        )
        .explode("match_id", empty_as_null=True)
        .select(
            pl.col("source1_entity_id").alias("s1_id"),
            "match_id",
            pl.col("match_id").str.slice(0, 2).alias("match_source"),
        )
    )


def convert(raw_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        for src in SOURCES:
            df = read_source_tsv(raw_dir / split / f"{split}_{src}.tsv")
            df.write_parquet(out_dir / f"{split}_{src}.parquet", compression="zstd")
            print(f"{split}_{src}: {df.height:,} rows")
    gt = read_ground_truth_tsv(raw_dir / "train" / "train_ground_truth.tsv")
    gt.write_parquet(out_dir / "train_gt_pairs.parquet", compression="zstd")
    print(f"train_gt_pairs: {gt.height:,} rows, {gt['s1_id'].n_unique():,} S1 entities")


def load_source(processed_dir: Path, split: str, src: str) -> pl.DataFrame:
    return pl.read_parquet(Path(processed_dir) / f"{split}_{src}.parquet")


def load_gt_pairs(processed_dir: Path) -> pl.DataFrame:
    return pl.read_parquet(Path(processed_dir) / "train_gt_pairs.parquet")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    convert(args.raw_dir, args.out_dir)
