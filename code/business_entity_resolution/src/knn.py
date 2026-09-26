"""Merge Kaggle embedding neighbours (Step 5.5 / 10.1) into the pipeline.

knn_{split}.parquet from kaggle/embed_knn: s1 (u32), m (u32), src (u8: 2|3), cos (f16) - top-6 per S1 per source.
  * recall_gain(): how many true val pairs the neighbours add beyond blocking
  * build_union(): candidate dir = blocking parts + neighbour-only pairs (<country>_knn.parquet, blocking cols null)
  * knn_table():   (s1, m, src, knn_cos, knn_rank) used as features for every candidate the embeddings found

    python -m src.knn --knn ../../artifacts/kaggle_out/knn_train.parquet --split train \
        --cands ../../artifacts/blocking/cands_all_df3000_t20n10a10 --out ../../artifacts/blocking/union_train
"""
import argparse
import os
import shutil
from pathlib import Path

import polars as pl

from .features import _num

CAND_SCHEMA = {"s1": pl.UInt32, "m": pl.UInt32, "src": pl.UInt8, "score": pl.Float32, "name_score": pl.Float32,
               "addr_score": pl.Float32, "rank": pl.UInt16, "rank_name": pl.UInt16, "rank_addr": pl.UInt16}


def knn_table(path: Path) -> pl.DataFrame:
    k = pl.read_parquet(path).with_columns(pl.col("cos").cast(pl.Float32).alias("knn_cos")).drop("cos")
    return k.with_columns(pl.col("knn_cos").rank("ordinal", descending=True).over("s1", "src").cast(pl.UInt8).alias("knn_rank"))


def recall_gain(knn: pl.DataFrame, cand_dir: Path, gt: pl.DataFrame, s1_ids: pl.Series) -> dict:
    ids = s1_ids.str.slice(3).cast(pl.UInt32).implode()
    t = (gt.drop_nulls("match_id").select(_num("s1_id").alias("s1"), pl.col("match_id").str.slice(1, 1).cast(pl.UInt8).alias("src"),
                                         _num("match_id").alias("m")).filter(pl.col("s1").is_in(ids)))
    b = pl.scan_parquet(Path(cand_dir) / "*.parquet").select("s1", "m", "src").filter(pl.col("s1").is_in(ids)).collect()
    k = knn.select("s1", "m", "src").filter(pl.col("s1").is_in(ids))
    in_b = t.join(b, on=["s1", "m", "src"], how="semi").height
    in_k = t.join(k, on=["s1", "m", "src"], how="semi").height
    in_u = t.join(pl.concat([b, k]).unique(), on=["s1", "m", "src"], how="semi").height
    new = k.join(b, on=["s1", "m", "src"], how="anti").height
    return {"true_pairs": t.height, "blocking_recall": in_b / t.height, "knn_recall": in_k / t.height,
            "union_recall": in_u / t.height, "knn_only_pairs_per_s1": new / s1_ids.len()}


def build_union(knn: pl.DataFrame, cand_dir: Path, norm_dir: Path, split: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.parquet"):
        old.unlink()
    for p in Path(cand_dir).glob("*.parquet"):  # hard links: no extra disk for the (read-only) blocking parts
        try:
            os.link(p, out_dir / p.name)
        except OSError:
            shutil.copyfile(p, out_dir / p.name)
    country = (pl.scan_parquet(norm_dir / f"{split}_source1.parquet").select(_num("entity_id").alias("s1"), "country").collect())
    blk = pl.scan_parquet(Path(cand_dir) / "*.parquet").select("s1", "m", "src").collect()
    extra = knn.select("s1", "m", "src").join(blk, on=["s1", "m", "src"], how="anti").join(country, on="s1")
    for (c,), g in extra.group_by("country"):
        g = g.drop("country").with_columns([pl.lit(None, dtype=t).alias(n) for n, t in CAND_SCHEMA.items() if n not in ("s1", "m", "src")])
        g.select(list(CAND_SCHEMA)).cast(CAND_SCHEMA).write_parquet(out_dir / f"{c}_knn.parquet")
        print(f"  {c}: +{g.height:,} neighbour-only pairs")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--knn", type=Path, required=True)
    ap.add_argument("--split", choices=["train", "test"], required=True)
    ap.add_argument("--cands", type=Path, required=True)
    ap.add_argument("--norm", type=Path, default=Path("../../artifacts/normalized"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--gt", type=Path, default=Path("../../artifacts/processed/train_gt_pairs.parquet"))
    ap.add_argument("--split-file", type=Path, default=Path("../../artifacts/splits/split_v1.parquet"))
    a = ap.parse_args()
    k = knn_table(a.knn)
    if a.split == "train":
        val = pl.read_parquet(a.split_file).filter(pl.col("role") == "val")["s1_id"]
        print("val recall:", {x: round(y, 4) for x, y in recall_gain(k, a.cands, pl.read_parquet(a.gt), val).items()})
    build_union(k, a.cands, a.norm, a.split, a.out)
