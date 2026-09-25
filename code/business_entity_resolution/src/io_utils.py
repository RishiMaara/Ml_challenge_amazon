"""Loading the challenge TSVs exactly as specified, and writing submission files.

Raw TSVs are never modified or converted: they are read with sep='\\t' and QUOTE_NONE
(addresses contain commas and quotes), and every derived table is saved as Parquet in
work_dir. Submission files are written as TSV, as the rules require.
"""

import csv
import json
import os
import time

import pandas as pd

SOURCE_COLS = ["entity_id", "business_name", "business_address", "country"]


def read_tsv(path):
    return pd.read_csv(
        path, sep="\t", dtype=str, keep_default_na=False, na_values=[],
        quoting=csv.QUOTE_NONE, encoding="utf-8", on_bad_lines="warn",
    )


def load_split(cfg, split):
    """Return one DataFrame with all three sources of a split plus a `source` column."""
    frames = []
    for s in (1, 2, 3):
        path = os.path.join(cfg.data_dir, split, f"{split}_source{s}.tsv")
        df = read_tsv(path)
        missing = [c for c in SOURCE_COLS if c not in df.columns]
        if missing:
            raise SystemExit(f"{path}: missing columns {missing}; found {list(df.columns)}")
        df = df[SOURCE_COLS].copy()
        df["source"] = s
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    for c in ("business_name", "business_address", "country"):
        out[c] = out[c].fillna("").astype(str).str.strip()
    out["entity_id"] = out["entity_id"].str.strip()
    return out


def load_ground_truth(cfg):
    """Return (gt_pairs DataFrame[s1_id, cand_id], set of all train S1 ids in the GT file)."""
    path = os.path.join(cfg.data_dir, "train", "train_ground_truth.tsv")
    gt = read_tsv(path)
    rows = []
    for s1, ids in zip(gt["source1_entity_id"].str.strip(), gt["matched_entity_ids"].fillna("")):
        for mid in ids.split(","):
            mid = mid.strip()
            if mid:
                rows.append((s1, mid))
    pairs = pd.DataFrame(rows, columns=["s1_id", "cand_id"]).drop_duplicates()
    s1_all = set(gt["source1_entity_id"].str.strip())
    if getattr(cfg, "sample_s1", 0) > 0:  # smoke run: only the sampled S1 entities
        from .profile_data import normalized_records
        recs = normalized_records(cfg, "train")
        keep = set(recs.loc[recs["source"] == 1, "entity_id"])
        pairs = pairs[pairs["s1_id"].isin(keep)]
        s1_all &= keep
    return pairs, s1_all


def gt_dict(gt_pairs):
    d = {}
    for s1, c in zip(gt_pairs["s1_id"], gt_pairs["cand_id"]):
        d.setdefault(s1, set()).add(c)
    return d


def write_id_lists(path, s1_ids, mapping, header):
    """Write one row per S1 id; ids joined by commas, empty when no matches."""
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write("\t".join(header) + "\n")
        for s1 in s1_ids:
            ids = mapping.get(s1, [])
            seen, uniq = set(), []
            for i in ids:
                if i not in seen:
                    seen.add(i)
                    uniq.append(i)
            f.write(f"{s1}\t{','.join(uniq)}\n")


def _clean(o):
    """Make any report JSON-safe: tuple/numpy keys -> str, numpy scalars -> Python."""
    import numpy as np
    if isinstance(o, dict):
        return {(k if isinstance(k, str) else str(k.item() if isinstance(k, np.generic) else k)): _clean(v)
                for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, float) and o != o:
        return None
    return o


def save_report(cfg, name, report):
    report = _clean(report)
    path = os.path.join(cfg.report_dir, f"{name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n=== {name} report (saved to {path}) ===")
    print(json.dumps(report, indent=2, default=str))


class Timer:
    def __init__(self, label):
        self.label = label

    def __enter__(self):
        self.t = time.time()
        print(f"[{time.strftime('%H:%M:%S')}] {self.label} ...", flush=True)
        return self

    def __exit__(self, *a):
        self.elapsed = time.time() - self.t
        print(f"[{time.strftime('%H:%M:%S')}] {self.label} done in {self.elapsed:.1f}s", flush=True)
