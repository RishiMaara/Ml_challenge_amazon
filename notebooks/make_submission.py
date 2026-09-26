"""Decision layer for v4+: one owner per record -> threshold -> matching_results.tsv (+ official validator).

    python notebooks/make_submission.py --scores artifacts/test_scores_v4.parquet --thr 0.75 --out submissions/v4
    python notebooks/make_submission.py --scores artifacts/test_scores_v4.parquet --ext artifacts/test_scores_ext.parquet ...

--ext: extra (s1, m, src, p) pairs from the blocking extension; they compete with the main pairs for ownership.
"""
import argparse
import subprocess
import sys
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code" / "business_entity_resolution"))
from src.evaluate import write_id_list_tsv  # noqa: E402
from src.train import one_owner, to_pairs  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--scores", type=Path, required=True)
ap.add_argument("--ext", type=Path, default=None)
ap.add_argument("--thr", type=float, default=0.75)
ap.add_argument("--ext-thr", type=float, default=None)
ap.add_argument("--out", type=Path, required=True)
a = ap.parse_args()

pred = pl.read_parquet(a.scores, columns=["s1", "m", "src", "p"]).with_columns(pl.lit(a.thr).alias("thr"))
if a.ext:
    ext = pl.read_parquet(a.ext, columns=["s1", "m", "src", "p"]).with_columns(pl.lit(a.ext_thr or a.thr).alias("thr"))
    pred = pl.concat([pred, ext.join(pred, on=["s1", "m", "src"], how="anti")])
sel = one_owner(pred).filter(pl.col("p") >= pl.col("thr"))
s1_all = pl.read_parquet(ROOT / "artifacts/normalized/test_source1.parquet", columns=["entity_id", "country"])
a.out.mkdir(parents=True, exist_ok=True)
write_id_list_tsv(to_pairs(sel), s1_all["entity_id"], a.out / "matching_results.tsv", "matched_entity_ids")
cty = s1_all.select(pl.col("entity_id").str.slice(3).cast(pl.UInt32).alias("s1"), "country")
st = sel.join(cty, on="s1").group_by("country").agg(pl.len().alias("pairs"), pl.col("s1").n_unique().alias("nonempty"))
st = st.join(cty.group_by("country").len(), on="country").with_columns(
    (pl.col("pairs") / pl.col("len")).round(3).alias("per_s1"), (1 - pl.col("nonempty") / pl.col("len")).round(4).alias("empty"))
print(st.sort("country"))
print(f"{sel.height:,} pairs written to {a.out / 'matching_results.tsv'}")
r = subprocess.run([sys.executable, "utils/validate_submission.py", "--matching", str((a.out / "matching_results.tsv").resolve()),
                    "--test-dir", "dataset/test"], cwd=ROOT / "student_resource", capture_output=True, text=True)
print(r.stdout[-800:], r.stderr[-800:])
