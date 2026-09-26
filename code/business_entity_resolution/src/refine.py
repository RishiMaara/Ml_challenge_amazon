"""Stage 3 (v4): sibling-aware refiner on top of the v3 matcher.

Error analysis (25 Sep) showed the remaining false merges are generator "siblings": a distractor derived from an S1
entity by perturbing the name (extra descriptor word, one word swapped, legal form changed) and shifting the house
number by a small positive amount on the same street (+1..+21). True copies instead carry digit typos (1630 -> 163,
683 -> 682) or a totally different number. The test set has ~2x more distractors per S1 than training
(5.76 vs 4.67 records per S1 at the same 3.46 matches per S1), so separating siblings matters even more there.

New, country-agnostic features on the pairs v3 already scores (p_v3 >= 0.02):
  house  : parsed house number (street-word / postcode / apartment aware), equality, signed numeric delta,
           edit distance, substring flag (old and new parse)
  name   : S1 name tokens missing from the record and record tokens absent from S1, after typo tolerance
  support: how many other pool records share the record's (name key, house) / name key (distractors are one-offs;
           true copies usually have siblings of their own), how many S1 share the record's / S1's name key
Model: LightGBM on v3 features + p_v3 + the new features, trained 2-fold on the validation S1 (p_v3 is out-of-sample
there); test = mean of the two fold models.
"""
import re
from pathlib import Path

import numpy as np
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein

from .normalize import to_ascii

STREET_WORDS = frozenset("""
rue r av ave avenue bd boulevard boul all allee chemin ch che pl place impasse imp route rte quai square sq cours crs
passage pass porte parvis promenade prom esplanade sentier voie villa cite hameau lieu lotissement residence res
mail rond point faubourg fbg montee traverse ruelle
st street rd road dr drive ln lane ct court blvd way pkwy parkway hwy highway trl trail cir circle loop ter terrace
pike pl plaza sq tpke turnpike aly alley row path run pass xing crossing cv cove pt point hl hill hts heights
main marg nagar road lane gali colony sector block cross
""".split())
APT_WORDS = frozenset("""appartement app appt apt unit suite ste batiment bat bldg building flat floor fl flr etage
etg bp cs po box pmb room rm dept escalier esc porte tour lot""".split())
SUFFIX_WORDS = frozenset({"bis", "ter", "quater", "b", "a", "c", "d", "t"})
ORD_WORDS = frozenset({"eme", "er", "e", "st", "nd", "rd", "th", "etage", "floor"})
_TOK = re.compile(r"[a-z]+|\d+")


def house_v2(raw: str | None, country: str) -> str:
    """House number: a number directly followed by a street word (optional bis/ter/letter in between); otherwise the
    first number that is not a postcode, an apartment/floor/box number or an ordinal."""
    if not raw:
        return ""
    parts = [_TOK.findall(p) for p in to_ascii(raw).lower().split(",")]
    for toks in parts:
        for i, t in enumerate(toks):
            if t.isdigit():
                j = i + 1
                while j < len(toks) and toks[j] in SUFFIX_WORDS and j + 1 < len(toks):
                    j += 1
                if j < len(toks) and toks[j] in STREET_WORDS and not (i > 0 and toks[i - 1] in APT_WORDS):
                    return t.lstrip("0") or "0"
    for toks in parts:
        for i, t in enumerate(toks):
            if not t.isdigit():
                continue
            nxt = toks[i + 1] if i + 1 < len(toks) else ""
            prv = toks[i - 1] if i > 0 else ""
            if prv in APT_WORDS or nxt in ORD_WORDS:
                continue
            if len(t) == 5 and (i == len(toks) - 1 or (i == 0 and nxt.isalpha() and nxt not in STREET_WORDS)):
                continue  # postcode (ends the part, or '59200 tourcoing')
            if len(t) == 6 and country == "India":
                continue  # PIN
            return t.lstrip("0") or "0"
    return ""


REC = ["name_core", "name_key", "name_alt", "name_norm", "addr_house_no"]


def records(norm_dir: Path, proc_dir: Path, split: str, source: int) -> pl.DataFrame:
    n = pl.read_parquet(norm_dir / f"{split}_source{source}.parquet", columns=["entity_id", "country"] + REC)
    raw = pl.read_parquet(proc_dir / f"{split}_source{source}.parquet", columns=["entity_id", "business_address"])
    n = n.join(raw, on="entity_id", how="left")
    hv = [house_v2(a, c) for a, c in zip(n["business_address"].to_list(), n["country"].to_list())]
    key = "s1" if source == 1 else "m"
    n = n.with_columns(pl.Series("house_v2", hv), pl.col("entity_id").str.slice(3).cast(pl.UInt32).alias(key)).drop(
        "business_address", "entity_id")
    if source != 1:
        n = n.with_columns(pl.lit(source, pl.UInt8).alias("src"))
    return n


def support_tables(norm_dir: Path, proc_dir: Path, split: str):
    """S1 table and pool table with label-free support counts."""
    s1 = records(norm_dir, proc_dir, split, 1)
    pool = pl.concat([records(norm_dir, proc_dir, split, s) for s in (2, 3)])
    kh = pool.filter(pl.col("house_v2") != "").group_by("country", "name_key", "house_v2").agg(pl.len().alias("_kh"))
    k = pool.group_by("country", "name_key").agg(pl.len().alias("_k"))
    s1k = s1.group_by("country", "name_key").agg(pl.len().alias("_s1k"))
    pool = (pool.join(kh, on=["country", "name_key", "house_v2"], how="left").join(k, on=["country", "name_key"], how="left")
                .join(s1k, on=["country", "name_key"], how="left")
                .with_columns((pl.col("_kh") - 1).cast(pl.Int32).alias("r_sup_kh"), (pl.col("_k") - 1).cast(pl.Int32).alias("r_sup_k"),
                              pl.col("_s1k").fill_null(0).cast(pl.Int32).alias("r_n_s1_key")).drop("_kh", "_k", "_s1k"))
    s1 = (s1.join(s1k, on=["country", "name_key"], how="left").join(kh, on=["country", "name_key", "house_v2"], how="left")
            .join(k, on=["country", "name_key"], how="left")
            .with_columns(pl.col("_s1k").cast(pl.Int32).alias("s1_n_key"), pl.col("_kh").fill_null(0).cast(pl.Int32).alias("s1_sup_kh"),
                          pl.col("_k").fill_null(0).cast(pl.Int32).alias("s1_sup_k")).drop("_s1k", "_kh", "_k"))
    return s1, pool


def _tok_found(t: str, others: list[str], concat: str) -> bool:
    if len(t) >= 3 and t in concat:
        return True
    return any(fuzz.ratio(t, o) >= 80 for o in others)


_STOP = frozenset({"du", "de", "des", "la", "le", "les", "d", "l", "and", "of", "the", "et", "en", "a", "pour", "sur", "aux", "au"})


def _initials(s: str) -> str:
    return "".join(t[0] for t in s.split() if t and t not in _STOP and t[0].isalpha())


def _acronym(n1_norm: str, n1_core: str, n2_norm: str) -> int:
    """Record name is the S1 name's initials (JM = Jerusalem Musique, LFP = Local Force Parents), in any order."""
    r = re.sub(r"[^a-z]", "", n2_norm)
    if not (2 <= len(r) <= 6) or " " in n2_norm.strip():
        return 0
    i, c = _initials(n1_norm), _initials(n1_core)
    return int(sorted(r) == sorted(c) or sorted(r) == sorted(i) or (len(c) >= 2 and set(c) <= set(r) and len(r) <= len(i)))


def _name_diff(c1: str, c2: str, alt2: str, df: dict) -> tuple:
    """Unmatched tokens both ways (typo tolerant) and the max corpus frequency of those tokens: a sibling swaps in a
    common vocabulary word (club, groupe), a rebrand/alias brings an invented rare one."""
    t1, t2 = c1.split(), c2.split()
    ta = alt2.split() if alt2 else []
    cat2 = (c2 + alt2).replace(" ", "")
    cat1 = c1.replace(" ", "")
    m1 = [t for t in t1 if not _tok_found(t, t2 + ta, cat2)]
    e2 = [t for t in t2 if not _tok_found(t, t1, cat1)]
    return (len(m1), len(e2), len(t1), len(t2), max((df.get(t, 0.0) for t in m1), default=-1.0),
            max((df.get(t, 0.0) for t in e2), default=-1.0))


def _int(s: pl.Expr) -> pl.Expr:
    return s.str.extract(r"^(\d{1,7})").cast(pl.Int64, strict=False)


def _house_feats(h1: str, h2: str, pre: str) -> list[pl.Expr]:
    both = (pl.col(h1) != "") & (pl.col(h2) != "")
    return [
        pl.when(both).then((pl.col(h1) == pl.col(h2)).cast(pl.Int8)).alias(f"{pre}_eq"),
        pl.when(both).then(_int(pl.col(h2)) - _int(pl.col(h1))).alias(f"{pre}_diff"),
        pl.when(both).then(pl.col(h1).str.contains(pl.col(h2), literal=True) | pl.col(h2).str.contains(pl.col(h1), literal=True))
          .cast(pl.Int8).alias(f"{pre}_sub"),
    ]


NEW_COLS = ["hv_eq", "hv_diff", "hv_sub", "hv_lev", "ho_diff", "ho_sub", "ho_lev", "hv_missing_1", "hv_missing_2",
            "nm_miss1", "nm_extra2", "nm_len1", "nm_len2", "nm_subst", "nm_miss1_df", "nm_extra2_df", "n_acr",
            "r_sup_kh", "r_sup_k", "r_n_s1_key", "s1_n_key", "s1_sup_kh", "s1_sup_k"]


def pair_new_features(pairs: pl.DataFrame, s1: pl.DataFrame, pool: pl.DataFrame) -> pl.DataFrame:
    """pairs: s1, m, src (+ anything). Returns pairs with NEW_COLS appended (same row order)."""
    a = s1.select("s1", "country", *[pl.col(c).alias(c + "_1") for c in REC + ["house_v2"]], "s1_n_key", "s1_sup_kh", "s1_sup_k")
    b = pool.select("m", "src", *[pl.col(c).alias(c + "_2") for c in REC + ["house_v2"]], "r_sup_kh", "r_sup_k", "r_n_s1_key")
    w = pairs.with_row_index("_i").join(a, on="s1", how="left").join(b, on=["m", "src"], how="left").sort("_i")
    w = w.with_columns([pl.col(c).fill_null("") for c in [x + s for x in REC + ["house_v2"] for s in ("_1", "_2")]])
    dfs = token_df(s1)
    nd = [_name_diff(x, y, z, dfs.get(c, {})) for x, y, z, c in
          zip(w["name_core_1"].to_list(), w["name_core_2"].to_list(), w["name_alt_2"].to_list(), w["country"].fill_null("").to_list())]
    nd = np.array(nd, dtype=np.float32).reshape(-1, 6)
    acr = [_acronym(x, y, z) for x, y, z in zip(w["name_norm_1"].to_list(), w["name_core_1"].to_list(), w["name_norm_2"].to_list())]
    hv_lev = [Levenshtein.distance(x, y) if x and y else -1 for x, y in zip(w["house_v2_1"].to_list(), w["house_v2_2"].to_list())]
    ho_lev = [Levenshtein.distance(x, y) if x and y else -1 for x, y in zip(w["addr_house_no_1"].to_list(), w["addr_house_no_2"].to_list())]
    w = w.with_columns(
        *_house_feats("house_v2_1", "house_v2_2", "hv"),
        *[e for e in _house_feats("addr_house_no_1", "addr_house_no_2", "ho") if not e.meta.output_name().endswith("_eq")],
        pl.Series("hv_lev", hv_lev, dtype=pl.Int16).replace(-1, None),
        pl.Series("ho_lev", ho_lev, dtype=pl.Int16).replace(-1, None),
        (pl.col("house_v2_1") == "").cast(pl.Int8).alias("hv_missing_1"),
        (pl.col("house_v2_2") == "").cast(pl.Int8).alias("hv_missing_2"),
        pl.Series("nm_miss1", nd[:, 0].astype(np.int16)), pl.Series("nm_extra2", nd[:, 1].astype(np.int16)),
        pl.Series("nm_len1", nd[:, 2].astype(np.int16)), pl.Series("nm_len2", nd[:, 3].astype(np.int16)),
        pl.Series("nm_subst", ((nd[:, 0] > 0) & (nd[:, 1] > 0)).astype(np.int8)),
        pl.Series("nm_miss1_df", nd[:, 4]), pl.Series("nm_extra2_df", nd[:, 5]), pl.Series("n_acr", acr, dtype=pl.Int8),
    )
    return pl.concat([pairs, w.select(NEW_COLS)], how="horizontal")


_DF_REF: dict = {}


def token_df(s1: pl.DataFrame) -> dict:
    """Per country: name_core token -> log2 bucket of the share of that country's S1 names containing it (per mille).
    Seen countries use the TRAIN S1 names as the single reference (identical values on train and test, so tree
    thresholds cannot flip on tiny split-to-split differences); unseen countries use their own S1 names.
    Coarse buckets keep the feature robust (bucket 0 = unseen/invented word)."""
    def build(df):
        t = (df.select("country", pl.col("name_core").str.split(" ").list.unique().alias("t")).explode("t")
               .filter(pl.col("t").str.len_chars() > 0).group_by("country", "t").len())
        n = df.group_by("country").len().rename({"len": "n"})
        t = t.join(n, on="country").with_columns(
            (1 + (1000 * pl.col("len") / pl.col("n") + 0.01).log(2).round(0).clip(-7, 10) + 7).alias("df"))
        return {c: dict(zip(g["t"].to_list(), g["df"].to_list())) for (c,), g in t.group_by("country")}
    if not _DF_REF:
        ref = Path(__file__).resolve().parents[3] / "artifacts/refine/train_s1.parquet"
        _DF_REF.update(build(pl.read_parquet(ref, columns=["country", "name_core"])))
    own = build(s1.select("country", "name_core"))
    return {c: _DF_REF.get(c, own[c]) for c in own}


PARAMS3 = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=100, feature_fraction=0.8,
               bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, max_bin=255, num_threads=0, verbose=-1, seed=2026)
