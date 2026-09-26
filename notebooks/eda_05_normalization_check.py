"""Step 3 check: how well does normalisation make TRUE matched pairs agree? (sample of train pairs)"""
import sys
from pathlib import Path

import numpy as np
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist

ROOT = Path(__file__).resolve().parents[1]
N = ROOT / "artifacts" / "normalized"
P = ROOT / "artifacts" / "processed"

gt = pl.read_parquet(P / "train_gt_pairs.parquet").drop_nulls("match_id").sample(200_000, seed=0)
ids1 = gt["s1_id"].unique().implode()
ids2 = gt["match_id"].unique().implode()
s1 = pl.scan_parquet(N / "train_source1.parquet").filter(pl.col("entity_id").is_in(ids1)).collect()
s23 = pl.scan_parquet([N / f"train_source{i}.parquet" for i in (2, 3)]).filter(pl.col("entity_id").is_in(ids2)).collect()
raw1 = (pl.scan_parquet(P / "train_source1.parquet").filter(pl.col("entity_id").is_in(ids1))
          .select("entity_id", pl.col("business_name").alias("raw_name")).collect())
raw23 = (pl.scan_parquet([P / f"train_source{i}.parquet" for i in (2, 3)]).filter(pl.col("entity_id").is_in(ids2))
           .select("entity_id", pl.col("business_name").alias("raw_name")).collect())

a = s1.join(raw1, on="entity_id").rename(lambda c: c + "_1")
b = s23.join(raw23, on="entity_id").rename(lambda c: c + "_2")
d = gt.join(a, left_on="s1_id", right_on="entity_id_1").join(b, left_on="match_id", right_on="entity_id_2")


def rate(expr, filt=None):
    x = d if filt is None else d.filter(filt)
    return x.select(expr.mean()).item(), x.height


rows = []
for c in ("US", "India"):
    f = pl.col("country_1") == c
    b_nonempty = f & ~pl.col("addr_empty_2")
    rows.append({
        "country": c,
        "raw_name_lower_eq": rate(pl.col("raw_name_1").str.to_lowercase() == pl.col("raw_name_2").str.to_lowercase(), f)[0],
        "name_core_eq": rate(pl.col("name_core_1") == pl.col("name_core_2"), f)[0],
        "name_key_eq": rate(pl.col("name_key_1") == pl.col("name_key_2"), f)[0],
        "state_found_s1": rate(pl.col("addr_state_1") != "", f)[0],
        "state_found_s2": rate(pl.col("addr_state_2") != "", b_nonempty)[0],
        "state_eq|both": rate(pl.col("addr_state_1") == pl.col("addr_state_2"),
                              b_nonempty & (pl.col("addr_state_1") != "") & (pl.col("addr_state_2") != ""))[0],
        "house_eq|both": rate(pl.col("addr_house_no_1") == pl.col("addr_house_no_2"),
                              b_nonempty & (pl.col("addr_house_no_1") != "") & (pl.col("addr_house_no_2") != ""))[0],
        "house_missing_s2": rate(pl.col("addr_house_no_2") == "", b_nonempty & (pl.col("addr_house_no_1") != ""))[0],
    })
print(pl.DataFrame(rows).with_columns(pl.col(pl.Float64).round(3)))

# name similarity on core names for Indic-origin names vs Latin names
for lab, filt in [("indic names", pl.col("name_indic_2")), ("latin names", ~pl.col("name_indic_2"))]:
    x = d.filter((pl.col("country_1") == "India") & filt)
    ts = cpdist(x["name_core_1"].to_list(), x["name_core_2"].to_list(), scorer=fuzz.token_set_ratio, workers=-1)
    sk = cpdist(x["name_skel_1"].to_list(), x["name_skel_2"].to_list(), scorer=fuzz.token_set_ratio, workers=-1)
    print(f"India {lab} (n={x.height}): core token_set p10/p25/p50 = {np.percentile(ts, [10, 25, 50]).round(0)}, "
          f"skeleton token_set p10/p25/p50 = {np.percentile(sk, [10, 25, 50]).round(0)}")

print("\nIndic-name examples after normalisation:")
for r in d.filter(pl.col("name_indic_2")).sample(12, seed=1).select("raw_name_2", "name_core_2", "name_core_1").iter_rows():
    print("  ", r)
print("\nState not found in S2/S3 (sample of non-empty addresses):")
x = d.filter(~pl.col("addr_empty_2") & (pl.col("addr_state_2") == "") & (pl.col("addr_state_1") != ""))
print(f"  count {x.height} of {d.height}")
for r in x.sample(min(10, x.height), seed=2).select("country_1", "addr_norm_2", "addr_state_1").iter_rows():
    print("  ", r)
