"""EDA part 1: structural facts (STATUS 2.1, 2.4, 2.5, 2.8, 2.9) on the full data."""
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code" / "business_entity_resolution"))
from src.data import load_gt_pairs, load_source  # noqa: E402

P = ROOT / "artifacts" / "processed"
pl.Config.set_tbl_rows(40)
pl.Config.set_fmt_str_lengths(90)
pl.Config.set_tbl_width_chars(200)


def hdr(t):
    print(f"\n{'=' * 8} {t} {'=' * 8}")


src = {(sp, s): load_source(P, sp, s) for sp in ("train", "test") for s in ("source1", "source2", "source3")}
gt = load_gt_pairs(P)

# ---- 2.1 duplicates ----
hdr("2.1 duplicates")
for (sp, s), df in src.items():
    dup_id = df.height - df["entity_id"].n_unique()
    dup_rec = df.height - df.select("business_name", "business_address", "country").unique().height
    print(f"{sp}_{s}: dup entity_id={dup_id:,}  dup (name,addr,country) rows={dup_rec:,}")

# ---- 2.4 one-S1-per-record ----
hdr("2.4 does each S2/S3 id map to <=1 S1?")
m = gt.drop_nulls("match_id")
per_rec = m.group_by("match_id").agg(pl.col("s1_id").n_unique().alias("n_s1"))
print(per_rec["n_s1"].value_counts().sort("n_s1"))
for s in ("source2", "source3"):
    ids = src[("train", s)]["entity_id"]
    matched = m.filter(pl.col("match_source") == ("S2" if s == "source2" else "S3"))["match_id"].unique()
    print(f"train_{s}: {ids.len():,} records, matched {matched.len():,} ({matched.len() / ids.len():.1%}), "
          f"unmatched {ids.len() - matched.len():,}; GT ids missing from file: "
          f"{(~matched.is_in(ids.implode())).sum()}")

# ---- 2.5 country agreement ----
hdr("2.5 do matched pairs share country?")
s1 = src[("train", "source1")].select(pl.col("entity_id").alias("s1_id"), pl.col("country").alias("c1"))
s23 = pl.concat([src[("train", "source2")], src[("train", "source3")]]).select(
    pl.col("entity_id").alias("match_id"), pl.col("country").alias("c2"))
mp = m.join(s1, on="s1_id").join(s23, on="match_id")
print(mp.group_by("c1", "c2").len().sort("len", descending=True))

# ---- matches per S1 by source & country ----
hdr("matches per S1 by country")
cnt = gt.group_by("s1_id").agg(
    pl.col("match_id").drop_nulls().len().alias("n"),
    (pl.col("match_source") == "S2").sum().alias("n_s2"),
    (pl.col("match_source") == "S3").sum().alias("n_s3"),
).join(s1, on="s1_id")
print(cnt.group_by("c1").agg(
    pl.len(), (pl.col("n") == 0).mean().alias("singleton_rate"), pl.col("n").mean().alias("mean_n"),
    (pl.col("n_s2") == 0).mean().alias("no_s2_rate"), (pl.col("n_s3") == 0).mean().alias("no_s3_rate"),
))

# ---- 2.8 source style ----
hdr("2.8 style by source/country (train + test)")
rows = []
for (sp, s), df in src.items():
    for c, g in df.group_by("country"):
        n, a = g["business_name"], g["business_address"]
        rows.append(dict(
            file=f"{sp}_{s}", country=c[0], n=g.height,
            name_upper=(n == n.str.to_uppercase()).mean(),
            addr_upper=((a == a.str.to_uppercase()) & (a != "")).mean(),
            addr_empty=(a == "").mean(),
            name_nonascii=n.str.contains(r"[^\x00-\x7F]").mean(),
            addr_nonascii=a.str.contains(r"[^\x00-\x7F]").mean(),
            has_5digit=a.str.contains(r"\b\d{5}\b").mean(),
            has_6digit=a.str.contains(r"\b\d{6}\b").mean(),
            name_url=n.str.contains(r"(?i)\.(com|in|net|org|fr|co)\b|www\.").mean(),
            name_junk=n.str.contains(r"^[^\p{L}\p{N}]|[#<>|~*]").mean(),
            name_len=n.str.len_chars().mean(), addr_len=a.str.len_chars().mean(),
            n_addr_parts=(a.str.count_matches(",") + 1).mean(),
        ))
print(pl.DataFrame(rows).sort("file", "country").with_columns(pl.col(pl.Float64).round(3)))

# ---- 2.9 script inventory ----
hdr("2.9 unicode scripts in non-ASCII chars (sample 300k rows per file)")


def script_of(ch):
    try:
        return unicodedata.name(ch).split(" ")[0]
    except ValueError:
        return "UNNAMED"


for (sp, s), df in src.items():
    samp = df.sample(min(300_000, df.height), seed=0)
    for col in ("business_name", "business_address"):
        c = Counter()
        for txt in samp[col].filter(samp[col].str.contains(r"[^\x00-\x7F]")).to_list():
            c.update({script_of(ch) for ch in txt if ord(ch) > 127})
        print(f"{sp}_{s}.{col}: {c.most_common(10)}")
