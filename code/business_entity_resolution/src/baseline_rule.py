"""Baseline (Step 6): exact rule on normalised fields.

match(s1, r)  <=>  same country, same name_key, same non-empty house number, and states agree (or r has no state).
A record matching several S1s is dropped (the one-S1-per-record constraint makes ties ambiguous).
For this rule baseline, candidate_pairs == matching pairs (the rule is the whole "model").

Usage:
    python -m src.baseline_rule --norm ../../artifacts/normalized --split test --out ../../output
    python -m src.baseline_rule --norm ../../artifacts/normalized --split train --eval-split ../../artifacts/splits/split_v1.parquet \
        --gt ../../artifacts/processed/train_gt_pairs.parquet
"""
import argparse
from pathlib import Path

import polars as pl

from .evaluate import evaluate, write_id_list_tsv

KEYS = ["country", "name_key", "addr_house_no"]


def rule_pairs(norm_dir: Path, split: str) -> tuple[pl.DataFrame, pl.Series]:
    cols = ["entity_id", "country", "name_key", "addr_house_no", "addr_state"]
    s1 = pl.scan_parquet(norm_dir / f"{split}_source1.parquet").select(cols)
    s23 = pl.scan_parquet([norm_dir / f"{split}_source{i}.parquet" for i in (2, 3)]).select(cols)
    pairs = (s1.filter(pl.col("addr_house_no") != "").rename({"entity_id": "s1_id", "addr_state": "st1"})
               .join(s23.rename({"entity_id": "match_id", "addr_state": "st2"}), on=KEYS)
               .filter((pl.col("st1") == pl.col("st2")) | (pl.col("st2") == ""))
               .select("s1_id", "match_id")
               .filter(pl.col("s1_id").count().over("match_id") == 1)   # drop records claimed by >1 S1
               .collect())
    all_s1 = pl.scan_parquet(norm_dir / f"{split}_source1.parquet").select("entity_id").collect()["entity_id"]
    return pairs, all_s1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--norm", type=Path, required=True)
    ap.add_argument("--split", choices=["train", "test"], default="test")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--eval-split", type=Path, default=None)
    ap.add_argument("--gt", type=Path, default=None)
    args = ap.parse_args()
    pairs, all_s1 = rule_pairs(args.norm, args.split)
    print(f"{args.split}: {pairs.height:,} matched pairs for {pairs['s1_id'].n_unique():,} of {all_s1.len():,} S1")
    if args.eval_split and args.gt:
        sp = pl.read_parquet(args.eval_split)
        for name, sub in [("val", sp.filter(pl.col("role") == "val")), ("dev", sp.filter(pl.col("is_dev")))]:
            res = evaluate(pairs, pl.read_parquet(args.gt), sub.select("s1_id", "country"))
            print(name, {k: round(v, 4) for k, v in res.items() if isinstance(v, float)})
            for c, v in res["by_country"].items():
                print("   ", c, {k: round(x, 4) for k, x in v.items() if isinstance(x, float)})
    if args.out:
        write_id_list_tsv(pairs, all_s1, args.out / "matching_results.tsv", "matched_entity_ids")
        write_id_list_tsv(pairs, all_s1, args.out / "candidate_pairs.tsv", "candidate_entity_ids")
        print("wrote", args.out / "matching_results.tsv", args.out / "candidate_pairs.tsv")


if __name__ == "__main__":
    main()
