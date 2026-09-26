"""Export compact text for GPU embedding on Kaggle (Step 5.5/10.1).

One parquet per split: id (u32), src (u8: 1|2|3), country (str), text = "<name_core>, <first locality>".
Only normalised challenge data is exported (no external data).

    python -m src.embed_export --norm ../../artifacts/normalized --out ../../artifacts/kaggle_export
"""
import argparse
from pathlib import Path

import polars as pl


def export(norm_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "test"):
        parts = []
        for src in (1, 2, 3):
            parts.append(
                pl.scan_parquet(norm_dir / f"{split}_source{src}.parquet")
                  .select(pl.col("entity_id").str.slice(3).cast(pl.UInt32).alias("id"),
                          pl.lit(src, pl.UInt8).alias("src"), "country",
                          pl.when(pl.col("addr_localities") != "")
                            .then(pl.col("name_core") + ", " + pl.col("addr_localities").str.split(" | ").list.first())
                            .otherwise(pl.col("name_core")).alias("text"))
                  .collect())
        df = pl.concat(parts)
        df.write_parquet(out_dir / f"{split}_text.parquet", compression="zstd", compression_level=10)
        print(split, df.height, f"{(out_dir / f'{split}_text.parquet').stat().st_size / 2**20:.0f} MB")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--norm", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    export(a.norm, a.out)
