"""Pair features for the matcher (Step 7).

Input : a candidate directory from src.blocking (s1, m, src, score, name_score, addr_score, rank, rank_name, rank_addr)
        + normalised records.
Output: parquet parts with one row per candidate pair: ids, label (train only), ~60 numeric features.

Feature groups
  name    : fuzzy similarities of cores / concatenations / skeletons / alias, key equality, legal-form agreement
  address : similarity of full address / street / localities, state / house number / postcode agreement, number overlap
  blocking: scores and ranks from the blocker
  context : within-S1 (score vs the S1's best candidate) and across-S1 (is this S1 the record's best owner?)
Country is deliberately NOT a feature (France is unseen in training).

Usage:
    python -m src.features --split train --cands ../../artifacts/blocking/cands_all_df3000_t20n10a10 \
        --subset dev --out ../../artifacts/features/dev
"""
import argparse
import time
from pathlib import Path

import numpy as np
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler
from rapidfuzz.process import cpdist

REC_COLS = ["name_core", "name_key", "name_skel", "name_legal", "name_alt", "name_is_domain", "name_indic",
            "addr_norm", "addr_street", "addr_localities", "addr_state", "addr_postcode", "addr_nums",
            "addr_house_no", "addr_empty"]


def _num(col: str) -> pl.Expr:
    return pl.col(col).str.slice(3).cast(pl.UInt32)


def load_records(norm_dir: Path, split: str, source: int, ids: pl.Series | None = None) -> pl.DataFrame:
    lf = pl.scan_parquet(norm_dir / f"{split}_source{source}.parquet").select(["entity_id"] + REC_COLS)
    if ids is not None:
        lf = lf.filter(_num("entity_id").is_in(ids.implode()))
    key = "s1" if source == 1 else "m"
    lf = lf.with_columns(_num("entity_id").alias(key)).drop("entity_id")
    if source != 1:
        lf = lf.with_columns(pl.lit(source, pl.UInt8).alias("src"))
    return lf.collect()


# ------------------------------------------------------------------ context features (numeric only)
def reverse_context(cand_dir: Path) -> pl.DataFrame:
    """Across-S1 competition per record (m, src): best / second-best score and name_score, #S1 claiming it."""
    lf = pl.scan_parquet(Path(cand_dir) / "*.parquet").select("m", "src", "score", "name_score")
    return lf.group_by("m", "src").agg(
        pl.len().cast(pl.UInt16).alias("rev_n_s1"),
        pl.col("score").max().alias("rev_best"),
        pl.col("score").top_k(2).min().alias("rev_second"),
        pl.col("name_score").max().alias("rev_best_name"),
    ).collect(engine="streaming")


def with_context(c: pl.DataFrame, rev: pl.DataFrame) -> pl.DataFrame:
    c = c.join(rev, on=["m", "src"], how="left")
    grp = ["s1", "src"]
    return c.with_columns(
        # within S1 (per source)
        (pl.col("score") / pl.col("score").max().over(grp)).alias("ctx_score_rel"),
        (pl.col("score") - pl.col("score").max().over(grp)).alias("ctx_score_gap"),
        (pl.col("name_score") / pl.col("name_score").max().over(grp).clip(lower_bound=1e-6)).alias("ctx_name_rel"),
        (pl.col("addr_score") / pl.col("addr_score").max().over(grp).clip(lower_bound=1e-6)).alias("ctx_addr_rel"),
        pl.len().over("s1").cast(pl.UInt16).alias("ctx_n_cands"),
        # across S1 for this record
        (pl.col("score") >= pl.col("rev_best")).alias("rev_is_best"),
        pl.when(pl.col("score") >= pl.col("rev_best")).then(pl.col("score") - pl.col("rev_second"))
          .otherwise(pl.col("score") - pl.col("rev_best")).alias("rev_margin"),
        (pl.col("score") / pl.col("rev_best")).alias("rev_score_rel"),
        (pl.col("name_score") / pl.col("rev_best_name").clip(lower_bound=1e-6)).alias("rev_name_rel"),
    ).drop("rev_best", "rev_second", "rev_best_name")


# ------------------------------------------------------------------ string features on a chunk
def _sim(a: list, b: list, scorer) -> np.ndarray:
    return cpdist(a, b, scorer=scorer, workers=-1, dtype=np.float32)


def _set_overlap(a: str, b: str, sep: str = " ") -> pl.Expr:
    """|A∩B| / |A∪B| for whitespace/sep-split sets, null when either side is empty."""
    A = pl.col(a).str.split(sep).list.eval(pl.element().filter(pl.element() != ""))
    B = pl.col(b).str.split(sep).list.eval(pl.element().filter(pl.element() != ""))
    inter = A.list.set_intersection(B).list.len()
    union = A.list.set_union(B).list.len()
    return pl.when((A.list.len() > 0) & (B.list.len() > 0)).then(inter / union).otherwise(None)


def _agree(a: str, b: str) -> pl.Expr:
    """1 = both present and equal, 0 = both present and different, null = a side missing."""
    return pl.when((pl.col(a) != "") & (pl.col(b) != "")).then((pl.col(a) == pl.col(b)).cast(pl.Int8)).otherwise(None)


def pair_features(df: pl.DataFrame) -> pl.DataFrame:
    """df has REC_COLS suffixed _1 (S1) and _2 (record). Returns numeric feature columns."""
    n1, n2 = df["name_core_1"].to_list(), df["name_core_2"].to_list()
    c1 = [x.replace(" ", "") for x in n1]
    c2 = [x.replace(" ", "") for x in n2]
    k1, k2 = df["name_skel_1"].to_list(), df["name_skel_2"].to_list()
    a1, a2 = df["addr_norm_1"].to_list(), df["addr_norm_2"].to_list()
    s1, s2 = df["addr_street_1"].to_list(), df["addr_street_2"].to_list()
    l1, l2 = df["addr_localities_1"].to_list(), df["addr_localities_2"].to_list()
    alt2 = df["name_alt_2"].to_list()
    f = {
        "n_ratio": _sim(n1, n2, fuzz.ratio),
        "n_partial": _sim(n1, n2, fuzz.partial_ratio),
        "n_tsort": _sim(n1, n2, fuzz.token_sort_ratio),
        "n_tset": _sim(n1, n2, fuzz.token_set_ratio),
        "n_jw": _sim(n1, n2, JaroWinkler.normalized_similarity),
        "n_concat_ratio": _sim(c1, c2, fuzz.ratio),
        "n_concat_partial": _sim(c1, c2, fuzz.partial_ratio),
        "n_skel_tset": _sim(k1, k2, fuzz.token_set_ratio),
        "n_alt_tset": _sim(n1, alt2, fuzz.token_set_ratio),
        "a_tset": _sim(a1, a2, fuzz.token_set_ratio),
        "a_partial": _sim(a1, a2, fuzz.partial_ratio),
        "a_street_tset": _sim(s1, s2, fuzz.token_set_ratio),
        "a_loc_tset": _sim(l1, l2, fuzz.token_set_ratio),
    }
    out = pl.DataFrame(f)
    ex = df.select(
        (pl.col("name_key_1") == pl.col("name_key_2")).cast(pl.Int8).alias("n_key_eq"),
        (pl.col("name_core_1") == pl.col("name_core_2")).cast(pl.Int8).alias("n_core_eq"),
        _set_overlap("name_core_1", "name_core_2").alias("n_tok_jacc"),
        (pl.col("name_core_1").str.split(" ").list.first() == pl.col("name_core_2").str.split(" ").list.first()).cast(pl.Int8).alias("n_first_eq"),
        (pl.col("name_core_1").str.count_matches(" ").cast(pl.Int16) - pl.col("name_core_2").str.count_matches(" ").cast(pl.Int16)).abs().alias("n_ntok_diff"),
        (pl.col("name_core_1").str.len_chars().cast(pl.Int32) - pl.col("name_core_2").str.len_chars().cast(pl.Int32)).abs().alias("n_len_diff"),
        _agree("name_legal_1", "name_legal_2").alias("n_legal_eq"),
        _set_overlap("name_legal_1", "name_legal_2").alias("n_legal_jacc"),
        (pl.col("name_legal_2") == "").cast(pl.Int8).alias("n_legal_missing_2"),
        (pl.col("name_alt_2") != "").cast(pl.Int8).alias("n_has_alt_2"),
        pl.col("name_is_domain_2").cast(pl.Int8).alias("n_is_domain_2"),
        pl.col("name_indic_2").cast(pl.Int8).alias("n_indic_2"),
        pl.col("addr_empty_2").cast(pl.Int8).alias("a_empty_2"),
        _agree("addr_state_1", "addr_state_2").alias("a_state_eq"),
        _agree("addr_house_no_1", "addr_house_no_2").alias("a_house_eq"),
        (pl.col("addr_house_no_2") == "").cast(pl.Int8).alias("a_house_missing_2"),
        _agree("addr_postcode_1", "addr_postcode_2").alias("a_postcode_eq"),
        _set_overlap("addr_nums_1", "addr_nums_2").alias("a_nums_jacc"),
        pl.col("addr_nums_1").str.split(" ").list.set_intersection(pl.col("addr_nums_2").str.split(" "))
          .list.eval(pl.element().filter(pl.element() != "")).list.len().cast(pl.Int8).alias("a_nums_shared"),
        _set_overlap("addr_localities_1", "addr_localities_2", " | ").alias("a_loc_jacc"),
        _set_overlap("addr_norm_1", "addr_norm_2").alias("a_tok_jacc"),
        (pl.col("addr_norm_1").str.len_chars().cast(pl.Int32) - pl.col("addr_norm_2").str.len_chars().cast(pl.Int32)).alias("a_len_diff"),
    )
    return pl.concat([out, ex], how="horizontal")


BLOCK_COLS = ["src", "score", "name_score", "addr_score", "rank", "rank_name", "rank_addr"]
CTX_COLS = ["ctx_score_rel", "ctx_score_gap", "ctx_name_rel", "ctx_addr_rel", "ctx_n_cands",
            "rev_n_s1", "rev_is_best", "rev_margin", "rev_score_rel", "rev_name_rel"]
CROWD_COLS = ["crowd_n_2", "crowd_names_2", "crowd_s1_n_1", "crowd_pool_n_1", "crowd_pool_names_1"]
_STREET_STOP = ["st", "ave", "rd", "blvd", "ln", "dr", "ct", "pl", "rue", "avenue", "boulevard", "allee", "chemin",
                "place", "route", "de", "du", "des", "la", "le", "les", "l", "d", "n", "s", "e", "w", "bis"]


# ------------------------------------------------------------------ v2: crowded addresses (France failure mode)
def _addr_sig() -> pl.Expr:
    """Exact address signature: country | house no. | sorted street words | first locality.
    Null unless a house number AND street words are present (bare numbers collide massively)."""
    words = (pl.col("addr_street").str.replace_all(r"\s*\|\s*", " ").str.split(" ")
               .list.eval(pl.element().filter(pl.element().str.contains(r"[a-z]") & (pl.element().str.len_chars() >= 2)))
               .list.set_difference(pl.lit(_STREET_STOP)).list.sort().list.join(" "))
    loc = pl.col("addr_localities").str.split(" | ").list.first().fill_null("")
    return pl.when((pl.col("addr_house_no") != "") & (words != "")).then(
        (pl.col("country") + "|" + pl.col("addr_house_no") + "|" + words + "|" + loc).hash(seed=11)).otherwise(None)


def address_crowding(norm_dir: Path, split: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    """How many records / distinct names share each exact address, on the pool (S2+S3) and S1 sides.
    A crowded address (several businesses in one building) means an address match says little."""
    cols = ["entity_id", "country", "addr_street", "addr_localities", "addr_house_no", "name_key"]
    pool = (pl.scan_parquet([norm_dir / f"{split}_source{i}.parquet" for i in (2, 3)]).select(cols)
              .with_columns(_addr_sig().alias("sig"), _num("entity_id").alias("m"),
                            pl.col("entity_id").str.slice(1, 1).cast(pl.UInt8).alias("src")).collect())
    sig_stats = pool.drop_nulls("sig").group_by("sig").agg(pl.len().cast(pl.UInt32).alias("n"),
                                                          pl.col("name_key").n_unique().cast(pl.UInt32).alias("names"))
    crowd_pool = (pool.select("m", "src", "sig").join(sig_stats, on="sig", how="left")
                      .select("m", "src", pl.col("sig").alias("sig_2"), pl.col("n").alias("crowd_n_2"),
                              pl.col("names").alias("crowd_names_2")))
    s1 = (pl.scan_parquet(norm_dir / f"{split}_source1.parquet").select(cols)
            .with_columns(_addr_sig().alias("sig"), _num("entity_id").alias("s1")).collect())
    s1_stats = s1.drop_nulls("sig").group_by("sig").agg(pl.len().cast(pl.UInt32).alias("crowd_s1_n_1"))
    crowd_s1 = (s1.select("s1", "sig").join(s1_stats, on="sig", how="left").join(sig_stats, on="sig", how="left")
                  .select("s1", pl.col("sig").alias("sig_1"), "crowd_s1_n_1", pl.col("n").alias("crowd_pool_n_1"),
                          pl.col("names").alias("crowd_pool_names_1")))
    return crowd_pool, crowd_s1


def v2_features(df: pl.DataFrame) -> pl.DataFrame:
    """Name-conflict and within-S1 name-rank features; df must hold whole S1 groups (chunks follow S1 boundaries)."""
    best = pl.max_horizontal("n_tset", "n_skel_tset", "n_alt_tset", "n_concat_partial")
    df = df.with_columns(best.alias("n_best"))
    grp = ["s1", "src"]
    return df.select(
        "n_best",
        ((pl.col("n_tok_jacc").fill_null(0) == 0) & (pl.col("n_skel_tset") < 50) & (pl.col("n_concat_partial") < 60))
          .cast(pl.Int8).alias("n_conflict"),
        (pl.col("n_best") / pl.col("n_best").max().over(grp).clip(lower_bound=1.0)).alias("ctx_nbest_rel"),
        (pl.col("n_best") >= pl.col("n_best").max().over(grp)).cast(pl.Int8).alias("ctx_is_best_name"),
        (pl.col("n_best").rank("dense", descending=True).over(grp)).cast(pl.UInt16).alias("ctx_name_rank"),
        pl.when(pl.col("sig_1").is_not_null() & pl.col("sig_2").is_not_null())
          .then((pl.col("sig_1") == pl.col("sig_2")).cast(pl.Int8)).otherwise(None).alias("a_sig_eq"),
    )


KNN_COLS = ["knn_cos", "knn_rank"]


def build(norm_dir: Path, split: str, cand_dir: Path, s1_nums: pl.Series | None, out_dir: Path,
          gt: pl.DataFrame | None = None, chunk: int = 1_000_000, log=print, knn: pl.DataFrame | None = None) -> None:
    """knn: optional (s1, m, src, knn_cos, knn_rank) from src.knn; adds embedding-similarity features (null if absent)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.parquet"):
        old.unlink()
    t0 = time.time()
    rev = reverse_context(cand_dir)
    crowd_pool, crowd_s1 = address_crowding(norm_dir, split)
    log(f"  reverse context: {rev.height:,} records; crowding tables ({time.time() - t0:.0f}s)")
    g = None
    if gt is not None:
        g = (gt.drop_nulls("match_id").select(_num("s1_id").alias("s1"), pl.col("match_id").str.slice(1, 1).cast(pl.UInt8).alias("src"),
                                            _num("match_id").alias("m")).with_columns(pl.lit(1, pl.Int8).alias("label")))
    # one country at a time (candidate parts are named <country>_<chunk>.parquet); context is country-local
    countries = sorted({p.name.rsplit("_", 1)[0] for p in Path(cand_dir).glob("*.parquet")})
    j = 0
    for country in countries:
        country_files = sorted(Path(cand_dir).glob(f"{country}_*.parquet"))
        log(f"  {country}: loading {len(country_files)} chunks and normalized records ({time.time() - t0:.0f}s)")
        
        lf1 = pl.scan_parquet(norm_dir / f"{split}_source1.parquet").filter(pl.col("country") == country)
        r1 = lf1.select(["entity_id"] + REC_COLS).with_columns(_num("entity_id").alias("s1")).drop("entity_id").collect().rename({c: c + "_1" for c in REC_COLS})
        
        lf23 = pl.scan_parquet([norm_dir / f"{split}_source2.parquet", norm_dir / f"{split}_source3.parquet"]).filter(pl.col("country") == country)
        r2 = lf23.select(["entity_id"] + REC_COLS).with_columns(_num("entity_id").alias("m"), pl.col("entity_id").str.slice(1,1).cast(pl.UInt8).alias("src")).drop("entity_id").collect().rename({c: c + "_2" for c in REC_COLS})

        keep = (["s1", "m"] + BLOCK_COLS + CTX_COLS + CROWD_COLS + (KNN_COLS if knn is not None else [])
                + (["label"] if g is not None else []))

        for cand_file in country_files:
            cands = pl.read_parquet(cand_file)
            if s1_nums is not None:
                cands = cands.filter(pl.col("s1").is_in(s1_nums.implode()))
            if cands.height == 0:
                continue

            cands = (with_context(cands, rev).join(crowd_pool, on=["m", "src"], how="left")
                     .join(crowd_s1, on="s1", how="left"))
            if g is not None:
                cands = cands.join(g, on=["s1", "m", "src"], how="left").with_columns(pl.col("label").fill_null(0))
            if knn is not None:
                cands = cands.join(knn.select(["s1", "m", "src"] + KNN_COLS), on=["s1", "m", "src"], how="left")

            wide = cands.join(r1, on="s1", how="left").join(r2, on=["m", "src"], how="left")
            base = pl.concat([wide.select(keep + ["sig_1", "sig_2"]), pair_features(wide)], how="horizontal_extend")
            pl.concat([base.drop("sig_1", "sig_2"), v2_features(base)], how="horizontal_extend").write_parquet(out_dir / f"part_{j:04d}.parquet")
            
            j += 1
            if j % 20 == 0:
                log(f"    {country} processed {j} chunks ({time.time() - t0:.0f}s)")
        del r1, r2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--norm", type=Path, default=Path("../../artifacts/normalized"))
    ap.add_argument("--split", choices=["train", "test"], required=True)
    ap.add_argument("--cands", type=Path, required=True)
    ap.add_argument("--split-file", type=Path, default=Path("../../artifacts/splits/split_v1.parquet"))
    ap.add_argument("--subset", choices=["dev", "val", "fit_sample", "fit_sample2", "all"], default="dev")
    ap.add_argument("--fit-sample", type=int, default=200_000)
    ap.add_argument("--gt", type=Path, default=Path("../../artifacts/processed/train_gt_pairs.parquet"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--knn", type=Path, default=None, help="knn_{split}.parquet from the Kaggle job")
    args = ap.parse_args()
    ids, gt = None, None
    if args.split == "train":
        sp = pl.read_parquet(args.split_file)
        if args.subset == "dev":
            sp = sp.filter(pl.col("is_dev"))
        elif args.subset == "val":
            sp = sp.filter(pl.col("role") == "val")
        elif args.subset == "fit_sample":
            sp = sp.filter(pl.col("role") == "fit").sample(args.fit_sample, seed=2026)
        elif args.subset == "fit_sample2":  # fresh fit S1, disjoint from fit_sample (stage-1 never saw them)
            fit = sp.filter(pl.col("role") == "fit")
            used = fit.sample(200_000, seed=2026)["s1_id"].implode()
            sp = fit.filter(~pl.col("s1_id").is_in(used)).sample(args.fit_sample, seed=7)
        ids = sp["s1_id"].str.slice(3).cast(pl.UInt32)
        gt = pl.read_parquet(args.gt)
    knn = None
    if args.knn:
        from .knn import knn_table
        knn = knn_table(args.knn)
    build(args.norm, args.split, args.cands, ids, args.out, gt, knn=knn)


if __name__ == "__main__":
    main()
