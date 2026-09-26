"""Scoring: exact macro F0.5 (as the leaderboard defines it), blocking quality, submission IO, experiment log.

All functions work on *pair tables*: pl.DataFrame[s1_id, match_id]. An S1 with no rows = empty prediction.

Usage (score a submission file against train ground truth for the val or dev split):
    python -m src.evaluate --pred out/matching_results.tsv --gt ../../artifacts/processed/train_gt_pairs.parquet \
        --split ../../artifacts/splits/split_v1.parquet --subset val
"""
import argparse
import json
import time
from pathlib import Path

import polars as pl

BETA = 0.5


# ------------------------------------------------------------------ core metric
def entity_scores(pred: pl.DataFrame, truth: pl.DataFrame, s1_ids: pl.DataFrame, beta: float = BETA) -> pl.DataFrame:
    """Per-S1 precision/recall/F_beta. `s1_ids` (column s1_id, plus any extra columns kept for breakdowns)
    defines the evaluation set: every S1 in it is scored, including those absent from `pred`/`truth`."""
    keys = s1_ids["s1_id"].implode()
    pred = pred.select("s1_id", "match_id").filter(pl.col("s1_id").is_in(keys)).drop_nulls().unique()
    truth = truth.select("s1_id", "match_id").filter(pl.col("s1_id").is_in(keys)).drop_nulls().unique()
    n_pred = pred.group_by("s1_id").len("n_pred")
    n_true = truth.group_by("s1_id").len("n_true")
    tp = pred.join(truth, on=["s1_id", "match_id"]).group_by("s1_id").len("tp")
    b2 = beta * beta
    df = (s1_ids.join(n_pred, on="s1_id", how="left").join(n_true, on="s1_id", how="left").join(tp, on="s1_id", how="left")
                .with_columns(pl.col("n_pred", "n_true", "tp").fill_null(0).cast(pl.Int64)))
    p = pl.col("tp") / pl.col("n_pred")
    r = pl.col("tp") / pl.col("n_true")
    return df.with_columns(
        pl.when(pl.col("n_pred") > 0).then(p).otherwise(None).alias("precision"),
        pl.when(pl.col("n_true") > 0).then(r).otherwise(None).alias("recall"),
        pl.when((pl.col("n_true") == 0) & (pl.col("n_pred") == 0)).then(1.0)
          .when(pl.col("tp") == 0).then(0.0)
          .otherwise((1 + b2) * p * r / (b2 * p + r)).alias("f"),
    )


def summarize(scores: pl.DataFrame, by: list[str] | None = None) -> pl.DataFrame:
    aggs = [
        pl.len().alias("n_s1"),
        pl.col("f").mean().alias("f05"),
        pl.col("precision").mean().alias("macro_p"),         # over S1s with >=1 prediction
        pl.col("recall").mean().alias("macro_r"),            # over S1s with >=1 true match
        (pl.col("tp").sum() / pl.col("n_pred").sum()).alias("micro_p"),
        (pl.col("tp").sum() / pl.col("n_true").sum()).alias("micro_r"),
        pl.col("f").filter(pl.col("n_true") == 0).mean().alias("singleton_acc"),
        (pl.col("n_pred") == 0).mean().alias("pred_empty_rate"),
    ]
    return (scores.group_by(by).agg(aggs).sort(by) if by else scores.select(aggs))


def evaluate(pred: pl.DataFrame, truth: pl.DataFrame, s1_ids: pl.DataFrame) -> dict:
    sc = entity_scores(pred, truth, s1_ids)
    out = summarize(sc).to_dicts()[0]
    if "country" in s1_ids.columns:
        out["by_country"] = {r["country"]: r for r in summarize(sc, ["country"]).to_dicts()}
    return out


# ------------------------------------------------------------------ blocking quality
def blocking_report(cands: pl.DataFrame, truth: pl.DataFrame, s1_ids: pl.DataFrame, pool_sizes: dict | None = None) -> dict:
    """Recall ceiling of a candidate set.
    pair_recall      : share of true (s1, match) pairs present in the candidates
    entity_full_rec  : share of non-singleton S1s whose every true match is a candidate
    oracle_f05       : F0.5 if the matcher were perfect on these candidates (pred = truth ∩ cands)
    reduction_ratio  : 1 - |cands| / sum_country(|S1_c| * |S2∪S3 pool_c|)   (needs pool_sizes)
    """
    keys = s1_ids["s1_id"].implode()
    c = cands.select("s1_id", "match_id").filter(pl.col("s1_id").is_in(keys)).drop_nulls().unique()
    t = truth.select("s1_id", "match_id").filter(pl.col("s1_id").is_in(keys)).drop_nulls().unique()
    hit = t.join(c, on=["s1_id", "match_id"])
    per = (t.group_by("s1_id").len("n_true")
             .join(hit.group_by("s1_id").len("n_hit"), on="s1_id", how="left").with_columns(pl.col("n_hit").fill_null(0)))
    out = {
        "n_s1": s1_ids.height,
        "n_cand_pairs": c.height,
        "cands_per_s1": c.height / max(s1_ids.height, 1),
        "pair_recall": hit.height / max(t.height, 1),
        "entity_full_recall": (per["n_hit"] == per["n_true"]).mean(),
        "oracle_f05": evaluate(hit, t, s1_ids.select("s1_id"))["f05"],
    }
    if pool_sizes and "country" in s1_ids.columns:
        total = sum(n * pool_sizes.get(cty, 0) for cty, n in s1_ids.group_by("country").len().iter_rows())
        out["reduction_ratio"] = 1 - c.height / max(total, 1)
    if "country" in s1_ids.columns:
        cc = c.join(s1_ids.select("s1_id", "country"), on="s1_id")
        hc = hit.join(s1_ids.select("s1_id", "country"), on="s1_id").group_by("country").len("hit")
        tc = t.join(s1_ids.select("s1_id", "country"), on="s1_id").group_by("country").len("true")
        nc = s1_ids.group_by("country").len("n")
        out["by_country"] = {r["country"]: {"pair_recall": r["hit"] / r["true"], "cands_per_s1": r["cands"] / r["n"]}
                             for r in tc.join(hc, on="country").join(nc, on="country")
                                        .join(cc.group_by("country").len("cands"), on="country").to_dicts()}
    return out


# ------------------------------------------------------------------ submission IO
def read_id_list_tsv(path: Path) -> pl.DataFrame:
    """Read matching_results.tsv / candidate_pairs.tsv / ground truth TSV into a pair table."""
    df = pl.read_csv(path, separator="\t", quote_char=None, infer_schema=False)
    col = df.columns[1]
    return (df.rename({df.columns[0]: "s1_id"})
              .with_columns(pl.col(col).fill_null("").str.split(",").alias("match_id"))
              .explode("match_id", empty_as_null=True).filter(pl.col("match_id") != "").select("s1_id", "match_id"))


def write_id_list_tsv(pairs: pl.DataFrame, s1_ids: list[str] | pl.Series, path: Path, col: str = "matched_entity_ids") -> None:
    """Write one row per S1 (in s1_ids order), comma-joined unique ids, empty for no matches."""
    agg = pairs.select("s1_id", "match_id").drop_nulls().unique().sort("s1_id", "match_id").group_by("s1_id", maintain_order=True).agg(
        pl.col("match_id").str.join(",").alias(col))
    out = (pl.DataFrame({"source1_entity_id": pl.Series(s1_ids, dtype=pl.Utf8)})
             .join(agg.rename({"s1_id": "source1_entity_id"}), on="source1_entity_id", how="left", maintain_order="left")
             .with_columns(pl.col(col).fill_null("")))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    out.write_csv(path, separator="\t", quote_style="never")


# ------------------------------------------------------------------ experiment log
def log_experiment(log_path: Path, name: str, metrics: dict, notes: str = "") -> None:
    rec = {"time": time.strftime("%Y-%m-%d %H:%M"), "name": name, "notes": notes, **metrics}
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, default=float) + "\n")


def render_log(log_path: Path, md_path: Path) -> None:
    """Render the JSONL experiment log as a markdown table (docs/experiments.md)."""
    recs = [json.loads(line) for line in open(log_path, encoding="utf-8")]
    cols = ["time", "name", "f05", "macro_p", "macro_r", "singleton_acc", "pair_recall", "oracle_f05", "cands_per_s1", "notes"]
    fmt = lambda v: f"{v:.4f}" if isinstance(v, float) and v == v else ("" if v is None or v != v else str(v))  # noqa: E731
    lines = ["# Experiment log", "", "Auto-generated from `artifacts/experiments.jsonl` by `src.evaluate.render_log`.", "",
             "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(fmt(r.get(c)) for c in cols) + " |" for r in recs]
    Path(md_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", type=Path, required=True)
    ap.add_argument("--gt", type=Path, required=True, help="train_gt_pairs.parquet")
    ap.add_argument("--split", type=Path, required=True)
    ap.add_argument("--subset", choices=["val", "dev", "fit", "all"], default="val")
    args = ap.parse_args()
    split = pl.read_parquet(args.split)
    if args.subset == "dev":
        split = split.filter(pl.col("is_dev"))
    elif args.subset != "all":
        split = split.filter(pl.col("role") == args.subset)
    res = evaluate(read_id_list_tsv(args.pred), pl.read_parquet(args.gt), split.select("s1_id", "country"))
    print(json.dumps(res, indent=2, default=float))


if __name__ == "__main__":
    main()
