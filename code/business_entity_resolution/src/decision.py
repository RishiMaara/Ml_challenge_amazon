"""Decision layer (Step 9): turn pair probabilities into per-S1 match sets.

All strategies first apply the one-owner constraint (each record goes to its highest-p S1; EDA 2.4).
  threshold(t)      : keep owned pairs with p >= t
  expected_f(beta)  : per S1, sort owned candidates by p and pick the prefix k maximising the plug-in expected F-beta
                        E[F_k] ~= (1+b^2) * sum_{i<=k} p_i / (k + b^2 * sum_all p_i)
                      vs. the empty prediction, whose expected score is P(no true match) ~= prod(1 - p_i)
  hybrid(floor)     : expected_f, but never include pairs with p < floor
"""
import polars as pl

from .train import one_owner

B2 = 0.25  # beta^2 for F0.5


def threshold(pred: pl.DataFrame, t: float) -> pl.DataFrame:
    return one_owner(pred).filter(pl.col("p") >= t)


def expected_f(pred: pl.DataFrame, floor: float = 0.0, temp: float = 1.0) -> pl.DataFrame:
    """pred: s1, m, src, p (+ anything). Returns the selected (owned) pairs.
    temp != 1 sharpens/flattens probabilities (p^temp) as a crude calibration knob."""
    own = one_owner(pred).with_columns((pl.col("p") ** temp).alias("q"))
    own = own.sort(["s1", "q"], descending=[False, True]).with_columns(
        pl.col("q").cum_sum().over("s1").alias("cum"),
        pl.int_range(1, pl.len() + 1).over("s1").alias("k"),
        pl.col("q").sum().over("s1").alias("n_hat"),
        (1 - pl.col("q")).log().sum().over("s1").exp().alias("p_empty"),
    ).with_columns(((1 + B2) * pl.col("cum") / (pl.col("k") + B2 * pl.col("n_hat"))).alias("ef"))
    best = own.group_by("s1").agg(pl.col("ef").max().alias("ef_best"), pl.col("p_empty").first())
    own = own.join(best, on="s1").filter(pl.col("ef_best") > pl.col("p_empty"))
    k_best = own.filter(pl.col("ef") >= pl.col("ef_best")).group_by("s1").agg(pl.col("k").min().alias("k_best"))
    sel = own.join(k_best, on="s1").filter(pl.col("k") <= pl.col("k_best"))
    if floor > 0:
        sel = sel.filter(pl.col("p") >= floor)
    return sel.drop("q", "cum", "k", "n_hat", "p_empty", "ef", "ef_best", "k_best")
