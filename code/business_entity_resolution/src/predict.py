"""Inference + submission writing (Step 12).

    python -m src.predict --features ../../artifacts/features/test --model ../../artifacts/models/lgb_v1.txt \
        --norm ../../artifacts/normalized --out ../../output [--cands ../../artifacts/blocking/cands_test_... --write-candidates]

Scores every candidate pair, applies the decision layer
  (unseen-country safeguard -> one owner per record -> threshold),
and writes matching_results.tsv (one row per test S1, empty when no match). Optionally writes
candidate_pairs.tsv = every pair the model scored (the final-stage candidate set).
"""
import argparse
import json
import time
from pathlib import Path

import lightgbm as lgb
import polars as pl

from .evaluate import write_id_list_tsv
from .train import one_owner, to_pairs

KEEP_FLOOR = 0.02  # pairs below this can never be selected; dropping them keeps memory small


def unseen_country_safeguard(pred: pl.DataFrame, norm_dir: Path, split: str = "test") -> pl.DataFrame:
    """For S1 in countries absent from TRAINING, reject pairs whose house numbers conflict (a_house_eq == 0).
    Evidence (LB): in the unseen country, same-name 'sibling' businesses on the same street at a different house
    number are hard negatives the model over-merges. Seen countries are read from the training data (open set)."""
    seen = pl.read_parquet(norm_dir / "train_source1.parquet", columns=["country"])["country"].unique().implode()
    cty = (pl.read_parquet(norm_dir / f"{split}_source1.parquet", columns=["entity_id", "country"])
             .select(pl.col("entity_id").str.slice(3).cast(pl.UInt32).alias("s1"), "country"))
    p = pred.join(cty, on="s1", how="left")
    bad = ~pl.col("country").is_in(seen) & (pl.col("a_house_eq") == 0)
    print(f"  unseen-country safeguard: rejected {p.filter(bad).height:,} house-conflict pairs "
          f"in {p.filter(~pl.col('country').is_in(seen))['country'].unique().to_list()}")
    return p.filter(~bad).drop("country")


def score(feat_dir: Path, booster: lgb.Booster, feats: list[str], log=print) -> pl.DataFrame:
    out = []
    parts = sorted(Path(feat_dir).glob("*.parquet"))
    for i, p in enumerate(parts):
        df = pl.read_parquet(p)
        X = df.select([pl.col(c).cast(pl.Float32) for c in feats]).to_numpy()
        df = df.select("s1", "m", "src", "a_house_eq").with_columns(pl.Series("p", booster.predict(X), dtype=pl.Float32))
        out.append(df.filter(pl.col("p") >= KEEP_FLOOR))
        if i % 20 == 0:
            log(f"  scored {i + 1}/{len(parts)} parts")
    return pl.concat(out)


def write_candidates(cand_dir: Path, s1_all: pl.Series, path: Path) -> None:
    """Stream candidate_pairs.tsv country by country (93M pairs would not fit as strings at once)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    seen = set()
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for country in sorted({p.name.rsplit("_", 1)[0] for p in Path(cand_dir).glob("*.parquet")}):
            agg = (pl.scan_parquet(Path(cand_dir) / f"{country}_*.parquet").select("s1", "m", "src").unique()
                     .with_columns((pl.lit("S") + pl.col("src").cast(pl.Utf8) + "-" + pl.col("m").cast(pl.Utf8)).alias("mid"))
                     .group_by("s1").agg(pl.col("mid").sort().str.join(",")).collect())
            for s1, ids in agg.iter_rows():
                f.write(f"S1-{s1}\t{ids}\n")
                seen.add(f"S1-{s1}")
        for s in s1_all.to_list():  # S1 with no candidates at all
            if s not in seen:
                f.write(f"{s}\t\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=Path, required=True)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--norm", type=Path, default=Path("../../artifacts/normalized"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--cands", type=Path, default=None)
    ap.add_argument("--write-candidates", action="store_true")
    ap.add_argument("--no-safeguard", action="store_true", help="disable the unseen-country house-number safeguard")
    ap.add_argument("--scores-out", type=Path, default=None, help="save all pair scores (parquet) for analysis")
    args = ap.parse_args()
    t0 = time.time()
    meta = json.loads(args.model.with_suffix(".json").read_text())
    thr = args.threshold if args.threshold is not None else meta["threshold"]
    booster = lgb.Booster(model_file=str(args.model))
    pred = score(args.features, booster, meta["features"])
    if args.scores_out:
        pred.write_parquet(args.scores_out)
    if not args.no_safeguard:
        pred = unseen_country_safeguard(pred, args.norm)
    matched = one_owner(pred).filter(pl.col("p") >= thr)
    s1_all = pl.read_parquet(args.norm / "test_source1.parquet", columns=["entity_id"])["entity_id"]
    write_id_list_tsv(to_pairs(matched), s1_all, args.out / "matching_results.tsv", "matched_entity_ids")
    n_s1 = matched["s1"].n_unique()
    print(f"thr {thr}: {matched.height:,} pairs, {n_s1:,}/{s1_all.len():,} S1 non-empty "
          f"({1 - n_s1 / s1_all.len():.3f} empty) in {time.time() - t0:.0f}s")
    if args.write_candidates and args.cands:
        write_candidates(args.cands, s1_all, args.out / "candidate_pairs.tsv")
        print(f"candidate_pairs.tsv written ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
