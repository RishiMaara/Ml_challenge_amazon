"""Blocking extension (v5): address/name composite keys that recover pairs the main blocker's per-S1 top-k caps drop.

Val error analysis: 3.3% of true pairs never reach the matcher; many are easy (same house + street with a typo'd
name, or same name + street with the house number missing/garbled). Keys (per country, rare values only):
    KA  house_v2 | street word        KD  name word | street word        KE  house_v2 | name word
Each S1's new pairs (not already candidates) are ranked by a cheap name+address score and capped at `top` per S1.
"""
from pathlib import Path

import numpy as np
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist

GENERIC = frozenset("""st street rd road ave avenue dr drive ln lane ct court blvd way pl place rue de du des la le les
nagar main cross floor no the and of n s e w sector block colony phase city road marg gali near opp behind
allee chemin boulevard impasse route square quai cours bis ter""".split())


_CODE = r"\b(?:[a-z]{1,3}\s?-\s?)?\d+[a-z]?(?:\s?[-/]\s?[a-z0-9]+)+|\b[a-z]{1,3}\s?-\s?\d+[a-z]?\b|\b[a-z]{1,3}\d+[a-z0-9]*\b|\b\d+[a-z]{1,2}\b"


_BIGRAM = r"(?:^|\s)[a-z]{3,} [a-z]{3,}(?:\s|$)"


def _num(c):
    return pl.col(c).str.slice(3).cast(pl.UInt32)


def _words(col: str) -> pl.Expr:
    return (pl.col(col).str.replace_all(r"[|]", " ").str.split(" ")
              .list.eval(pl.element().filter(pl.element().str.contains(r"^[a-z]{3,}$")))
              .list.set_difference(pl.lit(list(GENERIC))).list.unique())


def load(norm_dir: Path, refine_dir: Path, split: str, source: int, country: str | None = None) -> pl.DataFrame:
    key = "s1" if source == 1 else "m"
    n = pl.scan_parquet(norm_dir / f"{split}_source{source}.parquet").select(
        "entity_id", "country", "name_core", "name_key", "addr_street", "addr_norm", "addr_localities", "addr_state")
    n = (n.filter(pl.col("country") == country) if country else n).collect()
    raw = (pl.scan_parquet(norm_dir.parent / "processed" / f"{split}_source{source}.parquet").select("entity_id", "business_address")
             .join(n.lazy().select("entity_id"), on="entity_id", how="semi").collect())
    n = n.join(raw, on="entity_id", how="left")
    n = n.with_columns(_num("entity_id").alias(key)).drop("entity_id")
    t = pl.read_parquet(refine_dir / f"{split}_{'s1' if source == 1 else 'pool'}.parquet",
                        columns=[key, "house_v2"] + ([] if source == 1 else ["src"]))
    if source != 1:
        t = t.filter(pl.col("src") == source)
        n = n.with_columns(pl.lit(source, pl.UInt8).alias("src"))
    return n.join(t, on=[key] + ([] if source == 1 else ["src"]), how="left").with_columns(
        _words("addr_street").alias("sw"), _words("name_core").alias("nw"), _words("addr_localities").alias("lw"),
        pl.col("business_address").fill_null("").str.to_lowercase().str.extract_all(_CODE)
          .list.eval(pl.element().str.replace_all(r"[-/.\s]", "")).list.eval(pl.element().filter(pl.element().str.len_chars() >= 3))
          .list.unique().alias("cw"),
        # adjacent rare address words (both parities so overlapping bigrams are covered): 'greenray apartments'
        pl.concat_list(pl.col("addr_norm").fill_null("").str.extract_all(_BIGRAM),
                       pl.col("addr_norm").fill_null("").str.replace(r"^\S+\s*", "").str.extract_all(_BIGRAM))
          .list.eval(pl.element().str.strip_chars())
          .list.eval(pl.element().filter(~pl.element().str.split(" ").list.eval(pl.element().is_in(list(GENERIC))).list.any()))
          .list.unique().alias("bw")).drop("business_address")


def _key_types(df: pl.DataFrame, ids: list[str]) -> list[pl.LazyFrame]:
    """One lazy frame per key type with (ids..., k: UInt64 hash of 'TYPE|value|value'). Call per country."""
    h, lf = pl.col("house_v2"), df.lazy()
    hk = lambda *parts: pl.concat_str(parts).hash(seed=17).alias("k")  # noqa: E731
    return [
        lf.filter(h != "").explode("sw").drop_nulls("sw").select(*ids, hk(pl.lit("A|"), h, pl.lit("|"), pl.col("sw"))),
        lf.explode("nw").drop_nulls("nw").explode("sw").drop_nulls("sw").select(*ids, hk(pl.lit("D|"), pl.col("nw"), pl.lit("|"), pl.col("sw"))),
        lf.filter(h != "").explode("nw").drop_nulls("nw").select(*ids, hk(pl.lit("E|"), h, pl.lit("|"), pl.col("nw"))),
        # Indian-style address codes (B-46, D-2/201, 1-8-506/2/A) with the state or a locality word; exact name with state
        lf.filter(pl.col("addr_state") != "").explode("cw").drop_nulls("cw").select(*ids, hk(pl.lit("I|"), pl.col("cw"), pl.lit("|"), pl.col("addr_state"))),
        lf.explode("cw").drop_nulls("cw").explode("lw").drop_nulls("lw").select(*ids, hk(pl.lit("J|"), pl.col("cw"), pl.lit("|"), pl.col("lw"))),
        lf.filter((pl.col("addr_state") != "") & (pl.col("name_key").str.len_chars() >= 4))
          .select(*ids, hk(pl.lit("N|"), pl.col("name_key"), pl.lit("|"), pl.col("addr_state"))),
        lf.explode("bw").drop_nulls("bw").select(*ids, hk(pl.lit("B|"), pl.col("bw"))),
    ]


def extension_pairs(s1: pl.DataFrame, pool: pl.DataFrame, existing: pl.DataFrame, top: int = 10,
                    max_df: int = 50, chunk: int = 100_000) -> pl.DataFrame:
    """s1/pool from load() for ONE country; existing = (s1, m, src) already candidates.
    Returns new (s1, m, src, xk, xs_name, xs_addr, xs). Keys with more than max_df pool records are dropped."""
    pks = []
    for sk, pk in zip(_key_types(s1, ["s1"]), _key_types(pool, ["m", "src"])):
        pk = pk.join(sk.select("k").unique(), on="k", how="semi")  # only keys some S1 also has
        dfc = pk.group_by("k").len().filter(pl.col("len") <= max_df).select("k")
        pks.append(pk.join(dfc, on="k", how="semi").unique().collect())
    pk = pl.concat(pks).unique()
    del pks
    out = []
    s1_ids = s1["s1"]
    for i in range(0, s1.height, chunk):
        part = s1.filter(pl.col("s1").is_in(s1_ids.slice(i, chunk).implode()))
        sk = pl.concat([f.collect() for f in _key_types(part, ["s1"])]).unique()
        pr = (sk.join(pk, on="k").group_by("s1", "m", "src").agg(pl.len().alias("xk"))
                .join(existing, on=["s1", "m", "src"], how="anti"))
        if pr.height == 0:
            continue
        w = (pr.join(part.select("s1", pl.col("name_core").alias("n1"), pl.col("addr_norm").alias("a1")), on="s1")
               .join(pool.select("m", "src", pl.col("name_core").alias("n2"), pl.col("addr_norm").alias("a2")), on=["m", "src"]))
        ns = cpdist(w["n1"].to_list(), w["n2"].to_list(), scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float32)
        ad = cpdist(w["a1"].fill_null("").to_list(), w["a2"].fill_null("").to_list(), scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float32)
        w = w.select("s1", "m", "src", "xk").with_columns(pl.Series("xs_name", ns), pl.Series("xs_addr", ad))
        w = w.with_columns((pl.col("xs_name") + pl.col("xs_addr") + 20 * pl.col("xk")).alias("xs"))
        w = w.sort("xs", descending=True).group_by("s1").head(top)
        out.append(w)
    return pl.concat(out) if out else pl.DataFrame()
