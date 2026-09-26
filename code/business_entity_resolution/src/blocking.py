"""Candidate generation (blocking): IDF-weighted token overlap, computed as sparse joins.

Every record becomes a bag of typed tokens (hashed to u64). Name-side types:
    w  name word            k  name word skeleton        nk  full sorted name key
    nb unordered word pair  nc concatenated name          g   5-grams of the concatenated name (domain-style names)
    wl name word|locality
Address-side types:
    a  address word         h  house number              hs  house no.|street word     hk  house no.|street skeleton
Scores: name_score = sum idf*weight over shared name-side tokens, addr_score likewise, total = sum.
Tokens with document frequency > max_df are dropped. Each S1 keeps, per source (S2/S3), the union of the
top-k_total by total, top-k_name by name_score and top-k_addr by addr_score. Everything runs per country.

Usage:
    python -m src.blocking --norm ../../artifacts/normalized --split-file ../../artifacts/splits/split_v1.parquet \
        --gt ../../artifacts/processed/train_gt_pairs.parquet --subset dev --out ../../artifacts/blocking
"""
import argparse
import time
from pathlib import Path

import polars as pl

# street-type / filler words that carry no identity (after normalisation)
ADDR_STOP = [
    "st", "ave", "rd", "blvd", "ln", "dr", "ct", "pl", "cir", "hwy", "pkwy", "trl", "ter", "sq", "n", "s", "e", "w",
    "ne", "nw", "se", "sw", "apt", "ste", "fl", "bldg", "rte", "no", "near", "opp", "behind", "and", "of", "the",
    "rue", "avenue", "boulevard", "allee", "chemin", "place", "impasse", "route", "de", "du", "des", "la", "le", "les",
    "l", "d", "et", "bis", "floor", "road", "street", "lane", "nagar", "colony", "sec", "ph", "po", "dist", "tal",
]
NAME_TYPES = {"w": 1.0, "k": 0.5, "nk": 2.0, "nb": 1.2, "nc": 2.0, "g": 0.35, "wl": 1.5}
ADDR_TYPES = {"a": 0.6, "h": 0.4, "hs": 1.5, "hk": 1.0}
TYPE_WEIGHT = {**NAME_TYPES, **ADDR_TYPES}
TYPE_CODE = {t: i for i, t in enumerate(TYPE_WEIGHT)}
NAME_CODES = [TYPE_CODE[t] for t in NAME_TYPES]
DEFAULTS = dict(max_df=3000, k_total=20, k_name=10, k_addr=10)
GRAM, MIN_CONCAT = 5, 9
NORM_COLS = ["entity_id", "country", "name_core", "name_key", "name_skel", "addr_norm", "addr_street",
             "addr_localities", "addr_house_no"]


def _words(col: str) -> pl.Expr:
    return pl.col(col).str.replace_all(r"\s*\|\s*", " ").str.split(" ").list.eval(pl.element().filter(pl.element() != ""))


def _skel(e: pl.Expr) -> pl.Expr:
    """Cheap vectorised skeleton: first char + remaining chars without vowels/h (mill, mll -> mll)."""
    return e.str.slice(0, 1) + e.str.slice(1).str.replace_all(r"[aeiouyh]", "")


def record_tokens(df: pl.LazyFrame, side: str) -> pl.LazyFrame:
    """(rid, tok, typ) rows. side='pool' emits 5-grams only for single-token (domain-style) names,
    side='query' emits them for every long concatenated name, so S1 'saint cloud youth' can reach 'saintcloudyouth'."""
    alpha = lambda e: e.filter(e.str.contains(r"[a-z]") & (e.str.len_chars() >= 2))  # noqa: E731
    stop = pl.lit(ADDR_STOP)
    base = df.select(
        "rid", "addr_house_no",
        _words("name_core").list.eval(alpha(pl.element())).alias("nw_ord"),
        _words("name_skel").list.eval(pl.element().filter(pl.element().str.len_chars() >= 3)).list.unique().alias("nk_w"),
        _words("addr_norm").list.eval(alpha(pl.element())).list.set_difference(stop).list.unique().alias("aw"),
        _words("addr_street").list.eval(alpha(pl.element())).list.set_difference(stop).list.unique().alias("sw"),
        pl.col("addr_localities").str.split(" | ").list.eval(pl.element().filter(pl.element() != "")).list.head(2).alias("loc"),
        pl.col("name_key"),
        pl.col("name_core").str.replace_all(" ", "").alias("concat"),
    ).with_columns(pl.col("nw_ord").list.unique().alias("nw"), pl.col("nw_ord").list.len().alias("n_words"))

    def part(frame, typ):  # hash immediately; typ stored as a small int code
        return (frame.drop_nulls("t").filter(pl.col("t") != "")
                     .select("rid", (pl.lit(typ + ":") + pl.col("t")).hash(seed=7).alias("tok"),
                             pl.lit(TYPE_CODE[typ], pl.UInt8).alias("typ")))

    has_h = pl.col("addr_house_no") != ""
    pairs_src = base.select("rid", pl.col("nw_ord").list.head(5).alias("x")).explode("x", empty_as_null=True).drop_nulls("x")
    long_concat = pl.col("concat").str.len_chars() >= MIN_CONCAT
    grams_src = base.filter(long_concat & (pl.col("n_words") == 1)) if side == "pool" else base.filter(long_concat)
    grams = (grams_src.select("rid", pl.concat_list([pl.col("concat").str.slice(i, GRAM) for i in range(0, 40)]).alias("g"))
                      .explode("g", empty_as_null=True).filter(pl.col("g").str.len_chars() == GRAM))

    parts = [
        part(base.select("rid", pl.col("nw").alias("t")).explode("t", empty_as_null=True), "w"),
        part(base.select("rid", pl.col("nk_w").alias("t")).explode("t", empty_as_null=True), "k"),
        part(base.select("rid", pl.col("name_key").alias("t")), "nk"),
        part(base.select("rid", pl.col("concat").alias("t")), "nc"),
        part(pairs_src.join(pairs_src, on="rid").filter(pl.col("x") < pl.col("x_right"))
                      .select("rid", (pl.col("x") + "|" + pl.col("x_right")).alias("t")), "nb"),
        part(grams.select("rid", pl.col("g").alias("t")), "g"),
        part(base.select("rid", "nw", "loc").explode("nw", empty_as_null=True).explode("loc", empty_as_null=True)
                 .select("rid", (pl.col("nw") + "|" + pl.col("loc")).alias("t")), "wl"),
        part(base.select("rid", pl.col("aw").alias("t")).explode("t", empty_as_null=True), "a"),
        part(base.filter(has_h).select("rid", pl.col("addr_house_no").alias("t")), "h"),
        part(base.filter(has_h).select("rid", "addr_house_no", "sw").explode("sw", empty_as_null=True)
                 .select("rid", (pl.col("addr_house_no") + "|" + pl.col("sw")).alias("t")), "hs"),
        part(base.filter(has_h).select("rid", "addr_house_no", "sw").explode("sw", empty_as_null=True)
                 .select("rid", (pl.col("addr_house_no") + "|" + _skel(pl.col("sw"))).alias("t")), "hk"),
    ]
    return pl.concat(parts).unique(["rid", "tok"])


def tokens_chunked(df: pl.DataFrame, side: str, chunk: int = 400_000) -> pl.DataFrame:
    """record_tokens over row slices: string intermediates stay small, only hashed u64 tokens accumulate."""
    cols = ["rid"] + NORM_COLS[2:]
    return pl.concat([record_tokens(df.slice(i, chunk).lazy().select(cols), side).collect()
                      for i in range(0, df.height, chunk)])


def build_index(pool: pl.DataFrame, max_df: int) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Postings (tok -> rid) for the pool, pruned to tokens with df <= max_df, plus per-token weight (idf * type weight)."""
    toks = tokens_chunked(pool, "pool")
    n = pool.height
    df = toks.group_by("tok", "typ").agg(pl.len().alias("df"))
    tw = pl.DataFrame({"typ": [TYPE_CODE[t] for t in TYPE_WEIGHT], "tw": list(TYPE_WEIGHT.values())},
                      schema={"typ": pl.UInt8, "tw": pl.Float64})
    weights = (df.filter(pl.col("df") <= max_df).join(tw, on="typ")
                 .select("tok", "typ", (pl.col("tw") * (pl.lit(n + 1) / pl.col("df")).log()).cast(pl.Float32).alias("wt")))
    postings = toks.join(weights.select("tok"), on="tok").select("tok", "rid")
    return postings, weights


def query(q: pl.DataFrame, postings: pl.DataFrame, weights: pl.DataFrame, pool_src: pl.Series,
          k_total: int, k_name: int, k_addr: int, chunk: int = 4000):
    """Yields, per chunk of query records and per source: union of top-k by total / name / address score."""
    qt = tokens_chunked(q, "query")
    qt = (qt.rename({"rid": "qid"}).drop("typ").join(weights, on="tok")
            .with_columns(pl.col("typ").is_in(NAME_CODES).alias("is_name")).drop("typ"))
    src = pl.DataFrame({"rid": pl.int_range(0, pool_src.len(), eager=True).cast(pl.UInt32), "src": pool_src})
    qids = qt["qid"].unique().sort()
    for i in range(0, qids.len(), chunk):
        part = qt.filter(pl.col("qid").is_in(qids.slice(i, chunk).implode()))
        yield (part.join(postings, on="tok")
                   .group_by("qid", "rid").agg(
                       pl.col("wt").filter(pl.col("is_name")).sum().alias("name_score"),
                       pl.col("wt").filter(~pl.col("is_name")).sum().alias("addr_score"))
                   .with_columns((pl.col("name_score") + pl.col("addr_score")).alias("score"))
                   .join(src, on="rid")
                   .with_columns(
                       pl.col("score").rank("ordinal", descending=True).over("qid", "src").cast(pl.UInt16).alias("rank"),
                       pl.col("name_score").rank("ordinal", descending=True).over("qid", "src").cast(pl.UInt16).alias("rank_name"),
                       pl.col("addr_score").rank("ordinal", descending=True).over("qid", "src").cast(pl.UInt16).alias("rank_addr"))
                   .filter((pl.col("rank") <= k_total) | (pl.col("rank_name") <= k_name) | (pl.col("rank_addr") <= k_addr)))


def _num(col: str) -> pl.Expr:
    return pl.col(col).str.slice(3).cast(pl.UInt32)


def generate(norm_dir: Path, split: str, s1_ids: pl.Series | None, max_df: int, k_total: int, k_name: int, k_addr: int,
             out_dir: Path, log=print) -> Path:
    """Candidates for S1 records of `split` (optionally restricted to s1_ids) against the full S2+S3 pool.
    Streams compact parquet parts to out_dir: s1 (u32), m (u32), src (u8: 2|3), scores, ranks."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.parquet"):
        old.unlink()
    s1 = pl.scan_parquet(norm_dir / f"{split}_source1.parquet").select(NORM_COLS)
    if s1_ids is not None:
        s1 = s1.filter(pl.col("entity_id").is_in(s1_ids.implode()))
    s1 = s1.collect()
    pool_lf = pl.scan_parquet([norm_dir / f"{split}_source{i}.parquet" for i in (2, 3)]).select(NORM_COLS)
    for country in s1["country"].unique().sort().to_list():
        t0 = time.time()
        pool = pool_lf.filter(pl.col("country") == country).collect().with_row_index("rid").with_columns(pl.col("rid").cast(pl.UInt32))
        q = s1.filter(pl.col("country") == country).with_row_index("rid").with_columns(pl.col("rid").cast(pl.UInt32))
        postings, weights = build_index(pool, max_df)
        t1 = time.time()
        qmap = q.select(pl.col("rid").alias("qid"), _num("entity_id").alias("s1"))
        pmap = pool.select("rid", _num("entity_id").alias("m"))
        n = 0
        for j, res in enumerate(query(q, postings, weights, pool["entity_id"].str.slice(1, 1).cast(pl.UInt8),
                                      k_total, k_name, k_addr)):
            part = (res.join(qmap, on="qid").join(pmap, on="rid")
                       .select("s1", "m", "src", "score", "name_score", "addr_score", "rank", "rank_name", "rank_addr"))
            part.write_parquet(out_dir / f"{country}_{j:05d}.parquet")
            n += part.height
            if j % 50 == 0:
                log(f"    {country} chunk {j}: {n:,} pairs so far, {time.time() - t1:.0f}s")
        log(f"  {country}: pool {pool.height:,}, queries {q.height:,}, postings {postings.height:,}, "
            f"cands {n:,} | index {t1 - t0:.0f}s, query {time.time() - t1:.0f}s")
        del pool, postings, weights
    return out_dir


def load_candidates(cand_dir: Path, s1_ids: pl.Series | None = None) -> pl.DataFrame:
    """Read streamed parts back with string ids (s1_id, match_id) for evaluation."""
    lf = pl.scan_parquet(Path(cand_dir) / "*.parquet")
    if s1_ids is not None:
        lf = lf.filter(pl.col("s1").is_in(s1_ids.str.slice(3).cast(pl.UInt32).implode()))
    return lf.with_columns(
        (pl.lit("S1-") + pl.col("s1").cast(pl.Utf8)).alias("s1_id"),
        (pl.lit("S") + pl.col("src").cast(pl.Utf8) + pl.lit("-") + pl.col("m").cast(pl.Utf8)).alias("match_id"),
    ).collect()


def main():
    from .evaluate import blocking_report, log_experiment
    ap = argparse.ArgumentParser()
    ap.add_argument("--norm", type=Path, required=True)
    ap.add_argument("--split-file", type=Path, required=True)
    ap.add_argument("--gt", type=Path, default=None)
    ap.add_argument("--subset", default="dev", choices=["dev", "val", "all", "test"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-df", type=int, default=DEFAULTS["max_df"])
    ap.add_argument("--k-total", type=int, default=DEFAULTS["k_total"])
    ap.add_argument("--k-name", type=int, default=DEFAULTS["k_name"])
    ap.add_argument("--k-addr", type=int, default=DEFAULTS["k_addr"])
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    split_name = "test" if args.subset == "test" else "train"
    ids = None
    if args.subset in ("dev", "val"):
        sp = pl.read_parquet(args.split_file)
        sp = sp.filter(pl.col("is_dev")) if args.subset == "dev" else sp.filter(pl.col("role") == "val")
        ids = sp["s1_id"]
    t = time.time()
    name = f"cands_{args.subset}_df{args.max_df}_t{args.k_total}n{args.k_name}a{args.k_addr}{args.tag}"
    cand_dir = generate(args.norm, split_name, ids, args.max_df, args.k_total, args.k_name, args.k_addr, args.out / name)
    print(f"{name}: done in {time.time() - t:.0f}s")
    if args.gt and split_name == "train":
        sp = pl.read_parquet(args.split_file)
        sp = sp.filter(pl.col("role") == "val") if args.subset in ("all", "val") else sp.filter(pl.col("is_dev"))
        cands = load_candidates(cand_dir, sp["s1_id"])
        rep = blocking_report(cands, pl.read_parquet(args.gt), sp.select("s1_id", "country"))
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rep.items()})
        log_experiment(args.out.parent / "experiments.jsonl", name, {k: v for k, v in rep.items() if k != "by_country"},
                       f"token-overlap blocking v2 max_df={args.max_df} k(total/name/addr)={args.k_total}/{args.k_name}/{args.k_addr} [{args.subset}]")


if __name__ == "__main__":
    main()
