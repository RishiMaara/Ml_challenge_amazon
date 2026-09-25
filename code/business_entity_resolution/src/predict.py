"""Stage 4: score test candidates, decode, write both submission TSVs, validate."""

import json
import os
import pickle
import subprocess
import sys

import numpy as np
import pandas as pd

from .decode import decode
from .io_utils import write_id_lists, save_report, Timer
from .profile_data import normalized_records
from .train import predict_model, _logit


def run(cfg):
    with open(os.path.join(cfg.work_dir, "models.pkl"), "rb") as fh:
        bundle = pickle.load(fh)
    with open(os.path.join(cfg.work_dir, "decode.json")) as fh:
        dec = json.load(fh)
    df = pd.read_parquet(os.path.join(cfg.work_dir, "feats_test.parquet"))
    X = df[bundle["features"]].astype(np.float32).values

    probs = {}
    with Timer("scoring test pairs"):
        for k in bundle["kinds"]:
            probs[k] = np.mean([predict_model(k, m, X) for m in bundle["models"][k]], axis=0)
    if dec["model"] == "stack":
        Z = np.column_stack([_logit(probs[k]) for k in bundle["kinds"]])
        p = bundle["stacker"].predict_proba(Z)[:, 1]
    else:
        p = probs[dec["model"]]
    pdf = pd.DataFrame({"s1_id": df["s1_id"].values, "cand_id": df["cand_id"].values, "p": p})
    pdf.to_parquet(os.path.join(cfg.work_dir, "test_scores.parquet"), index=False)

    pred = decode(pdf, dec["mode"], threshold=dec.get("threshold") or 0.5, bias=dec.get("bias", 0.0),
                  exclusive=bool(dec["exclusive"]), min_prob=cfg.min_prob, beta=cfg.beta)

    recs = normalized_records(cfg, "test").drop_duplicates("entity_id")
    s1 = recs[recs["source"] == 1]
    s1_ids = s1["entity_id"].tolist()
    cand_map = pdf.groupby("s1_id")["cand_id"].agg(list).to_dict()
    match_map = {s: sorted(v) for s, v in pred.items()}
    m_path = os.path.join(cfg.output_dir, "matching_results.tsv")
    c_path = os.path.join(cfg.output_dir, "candidate_pairs.tsv")
    write_id_lists(m_path, s1_ids, match_map, ["source1_entity_id", "matched_entity_ids"])
    write_id_lists(c_path, s1_ids, cand_map, ["source1_entity_id", "candidate_entity_ids"])

    t = pd.DataFrame({"s1": s1_ids, "country": s1["country"].values})
    t["n_pred"] = [len(pred.get(s, ())) for s in s1_ids]
    t["n_cand"] = [len(cand_map.get(s, ())) for s in s1_ids]
    pdf["country"] = s1.set_index("entity_id")["country"].reindex(pdf["s1_id"]).values
    rep = {
        "decode": dec,
        "s1_entities": len(s1_ids),
        "predicted_pairs": int(t["n_pred"].sum()),
        "by_country": {c: {"s1": int(len(g)), "pct_with_match": round(100 * float((g["n_pred"] > 0).mean()), 2),
                           "avg_matches": round(float(g["n_pred"].mean()), 3),
                           "avg_candidates": round(float(g["n_cand"].mean()), 2)}
                       for c, g in t.groupby("country")},
        "prob_quantiles_by_country": {c: np.round(np.percentile(g["p"], [50, 90, 99, 99.9]), 4).tolist()
                                      for c, g in pdf.groupby("country")},
    }
    save_report(cfg, "04_predict", rep)

    validator = os.path.abspath(os.path.join(cfg.data_dir, "..", "utils", "validate_submission.py"))
    if os.path.exists(validator):
        subprocess.run([sys.executable, validator, "--matching", m_path, "--candidate", c_path,
                        "--test-dir", os.path.join(cfg.data_dir, "test")], check=False)
    else:
        print(f"validator not found at {validator}; run student_resource/utils/validate_submission.py manually")
