"""Stage 0: normalize every record once (saved as Parquet) and print a data profile.

The profile answers the questions that decide the design: singleton rate, matches per
S1, whether an S2/S3 id ever belongs to two S1 entities (exclusivity), how often true
pairs cross countries, postcode coverage, and how noisy true-pair names really are.
"""

import os
import platform
from collections import Counter

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

from .io_utils import load_split, load_ground_truth, save_report, Timer
from .normalize import normalize_frame


def machine_info():
    info = {"python": platform.python_version(), "os": platform.platform(), "cpus": os.cpu_count()}
    try:
        import psutil
        info["ram_gb"] = round(psutil.virtual_memory().total / 2**30, 1)
    except Exception:
        info["ram_gb"] = "install psutil to report"
    return info


def normalized_records(cfg, split, sample=True):
    path = os.path.join(cfg.records_dir, f"records_{split}.parquet")
    if os.path.exists(path):
        df = pd.read_parquet(path)
    else:
        with Timer(f"load + normalize {split}"):
            df = normalize_frame(load_split(cfg, split))
            df.to_parquet(path, index=False)
    if sample and cfg.sample_s1 > 0:
        s1 = df[df["source"] == 1]
        keep = set(s1.sample(min(cfg.sample_s1, len(s1)), random_state=cfg.seed)["entity_id"])
        df = df[(df["source"] != 1) | df["entity_id"].isin(keep)].reset_index(drop=True)
    return df


def _q(x):
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return {}
    return {f"p{p}": round(float(np.percentile(x, p)), 3) for p in (1, 5, 10, 25, 50, 75, 90)}


def run(cfg):
    rep = {"machine": machine_info()}
    recs = {}
    for split in ("train", "test"):
        df = normalized_records(cfg, split, sample=False)
        recs[split] = df
        r = {}
        for s in (1, 2, 3):
            sub = df[df["source"] == s]
            r[f"source{s}"] = {
                "rows": int(len(sub)),
                "countries": sub["country"].value_counts().head(10).to_dict(),
                "empty_name_pct": round(100 * (sub["business_name"] == "").mean(), 2),
                "empty_addr_pct": round(100 * (sub["business_address"] == "").mean(), 2),
                "postcode_found_pct": round(100 * (sub["postcode"] != "").mean(), 2),
                "house_no_found_pct": round(100 * (sub["house_no"] != "").mean(), 2),
                "name_len_median": float(sub["business_name"].str.len().median()),
                "addr_len_median": float(sub["business_address"].str.len().median()),
                "dup_id_rows": int(sub["entity_id"].duplicated().sum()),
            }
        s1 = df[df["source"] == 1]
        core_counts = s1.groupby(["country_key", "name_core"]).size()
        r["s1_name_core_shared_pct"] = round(
            100 * float(core_counts[core_counts > 1].sum()) / max(1, len(s1)), 2)
        r["s1_top_repeated_names"] = core_counts.sort_values(ascending=False).head(10).to_dict()
        pc = df[df["postcode"] != ""].groupby(["country_key", "postcode"]).size()
        r["records_per_postcode"] = _q(pc.values)
        rep[split] = r

    # France (or any test-only country) samples: lets us check normalization rules
    train_c = set(recs["train"]["country_key"])
    new = recs["test"][~recs["test"]["country_key"].isin(train_c)]
    rep["test_only_countries"] = new["country"].value_counts().to_dict()
    rep["test_only_samples"] = (
        new.sample(min(15, len(new)), random_state=cfg.seed)[
            ["entity_id", "business_name", "business_address", "postcode", "name_core", "addr_norm"]]
        .to_dict("records") if len(new) else [])

    # ---------------- ground truth ----------------
    gt, gt_s1 = load_ground_truth(cfg)
    tr = recs["train"].drop_duplicates("entity_id").set_index("entity_id")
    s1_ids = set(tr.index[tr["source"] == 1])
    per_s1 = gt.groupby("s1_id").size()
    n_s1 = len(s1_ids)
    dist = Counter(min(int(per_s1.get(s, 0)), 6) for s in s1_ids)
    cand_owner = gt.groupby("cand_id")["s1_id"].nunique()
    g = {
        "s1_in_gt_file": len(gt_s1), "s1_in_source1": n_s1,
        "s1_missing_from_gt_file": len(s1_ids - gt_s1),
        "true_pairs": int(len(gt)),
        "singleton_pct": round(100 * dist[0] / max(1, n_s1), 2),
        "matches_per_s1_hist(6=6+)": {k: dist[k] for k in sorted(dist)},
        "pairs_to_s2": int(gt["cand_id"].str.startswith("S2-").sum()),
        "pairs_to_s3": int(gt["cand_id"].str.startswith("S3-").sum()),
        "cand_ids_owned_by_multiple_s1": int((cand_owner > 1).sum()),
        "gt_ids_not_in_sources": int((~gt["cand_id"].isin(tr.index)).sum()),
        "s2_rows_matched_pct": round(100 * gt["cand_id"].str.startswith("S2-").sum()
                                     / max(1, (tr["source"] == 2).sum()), 2),
        "s3_rows_matched_pct": round(100 * gt["cand_id"].str.startswith("S3-").sum()
                                     / max(1, (tr["source"] == 3).sum()), 2),
    }
    gt = gt[gt["cand_id"].isin(tr.index) & gt["s1_id"].isin(tr.index)]
    a = tr.loc[gt["s1_id"].values]
    b = tr.loc[gt["cand_id"].values]
    same_country = (a["country_key"].values == b["country_key"].values)
    g["true_pairs_same_country_pct"] = round(100 * same_country.mean(), 3)
    both_pc = (a["postcode"].values != "") & (b["postcode"].values != "")
    g["true_pairs_both_postcode_pct"] = round(100 * both_pc.mean(), 2)
    g["true_pairs_postcode_equal_given_both_pct"] = round(
        100 * (a["postcode"].values[both_pc] == b["postcode"].values[both_pc]).mean(), 2) if both_pc.any() else None
    both_h = (a["house_no"].values != "") & (b["house_no"].values != "")
    g["true_pairs_house_equal_given_both_pct"] = round(
        100 * (a["house_no"].values[both_h] == b["house_no"].values[both_h]).mean(), 2) if both_h.any() else None
    g["true_pairs_name_norm_equal_pct"] = round(100 * (a["name_norm"].values == b["name_norm"].values).mean(), 2)
    g["true_pairs_name_core_equal_pct"] = round(100 * (a["name_core"].values == b["name_core"].values).mean(), 2)
    tsr = [fuzz.token_set_ratio(x, y) for x, y in zip(a["name_core"].values, b["name_core"].values)]
    atsr = [fuzz.token_set_ratio(x, y) for x, y in zip(a["addr_norm"].values, b["addr_norm"].values)]
    g["true_pairs_name_token_set_ratio"] = _q(tsr)
    g["true_pairs_addr_token_set_ratio"] = _q(atsr)
    per_country = {}
    for c in sorted(set(a["country_key"])):
        m = a["country_key"].values == c
        per_country[c] = {"pairs": int(m.sum()),
                          "name_tsr_p10": float(np.percentile(np.array(tsr)[m], 10)),
                          "addr_tsr_p10": float(np.percentile(np.array(atsr)[m], 10))}
    g["per_country"] = per_country
    s = gt.sample(min(25, len(gt)), random_state=cfg.seed)
    g["true_pair_samples"] = [
        {"s1": f"{tr.at[x, 'business_name']} | {tr.at[x, 'business_address']}",
         "match": f"{tr.at[y, 'business_name']} | {tr.at[y, 'business_address']}"}
        for x, y in zip(s["s1_id"], s["cand_id"])]
    rep["ground_truth"] = g
    save_report(cfg, "00_profile", rep)
