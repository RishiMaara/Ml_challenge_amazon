"""Turning pair probabilities into per-S1 match sets, and the exact challenge metric.

The leaderboard metric is F0.5 per Source-1 entity, macro-averaged, with singletons
scoring 1.0 only for an empty prediction. So the right decision is per S1 entity, not per
pair: for each S1 we pick the set (empty, top-1, top-2, ...) with the highest *expected*
F0.5 under the model's probabilities. That lets one S1 with a single 0.55 candidate stay
empty (singleton risk) while another with 0.9/0.8 keeps both.
"""

import numpy as np
import pandas as pd


def macro_f05(pred, gt_d, s1_ids, beta=0.5, per_entity=False):
    b2 = beta * beta
    scores = []
    for s in s1_ids:
        g = gt_d.get(s, set())
        p = pred.get(s, set())
        if not g:
            scores.append(1.0 if not p else 0.0)
            continue
        tp = len(p & g)
        if tp == 0:
            scores.append(0.0)
            continue
        prec, rec = tp / len(p), tp / len(g)
        scores.append((1 + b2) * prec * rec / (b2 * prec + rec))
    arr = np.array(scores)
    return (arr.mean() if len(arr) else 0.0, arr) if per_entity else (arr.mean() if len(arr) else 0.0)


def _pb_pmf(ps):
    """Poisson-binomial pmf of the number of successes."""
    pmf = np.array([1.0])
    for p in ps:
        pmf = np.convolve(pmf, [1.0 - p, p])
    return pmf


def best_prefix(probs, beta=0.5, min_prob=0.02, max_k=10):
    """Return k (0 = predict empty) maximising expected F_beta for probs sorted descending."""
    b2 = beta * beta
    probs = np.asarray(probs, dtype=float)
    n = len(probs)
    if n == 0 or probs[0] < min_prob:
        return 0
    best_k, best_v = 0, float(np.prod(1.0 - probs))  # empty set is right iff nothing is true
    kmax = int(min(max_k, (probs >= min_prob).sum()))
    suffix = [None] * (n + 1)
    suffix[n] = np.array([1.0])
    for i in range(n - 1, -1, -1):
        suffix[i] = np.convolve(suffix[i + 1], [1.0 - probs[i], probs[i]])
    prefix = np.array([1.0])
    for k in range(1, kmax + 1):
        prefix = np.convolve(prefix, [1.0 - probs[k - 1], probs[k - 1]])
        a = np.arange(len(prefix))[:, None]
        b = np.arange(len(suffix[k]))[None, :]
        f = (1 + b2) * a / (b2 * (a + b) + k)
        v = float((prefix[:, None] * suffix[k][None, :] * f).sum())
        if v > best_v:
            best_k, best_v = k, v
    return best_k


def adjust(p, bias=0.0, temp=1.0):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    z = np.log(p / (1 - p)) / temp + bias
    return 1.0 / (1.0 + np.exp(-z))


def apply_exclusive(df, col="p"):
    """Each S2/S3 record belongs to at most one S1: zero out every non-best claim."""
    best = df.groupby("cand_id", sort=False)[col].transform("max")
    r = df.groupby("cand_id", sort=False)[col].rank(ascending=False, method="first")
    return np.where((df[col].values >= best.values) & (r.values == 1), df[col].values, 0.0)


def decode(df, mode="expected_f", threshold=0.5, bias=0.0, exclusive=True,
           min_prob=0.02, beta=0.5, max_k=10):
    """df columns: s1_id, cand_id, p. Returns {s1_id: set(cand_ids)}."""
    d = df[["s1_id", "cand_id", "p"]].copy()
    d["p"] = adjust(d["p"].values, bias=bias)
    if exclusive:
        d["p"] = apply_exclusive(d)
    if mode == "threshold":
        sel = d[d["p"] >= threshold]
        return sel.groupby("s1_id")["cand_id"].agg(set).to_dict()
    d = d[d["p"] >= 1e-3].sort_values(["s1_id", "p"], ascending=[True, False])
    out = {}
    s1 = d["s1_id"].values
    cid = d["cand_id"].values
    pv = d["p"].values
    bounds = np.flatnonzero(np.r_[True, s1[1:] != s1[:-1], True])
    for i in range(len(bounds) - 1):
        a, b = bounds[i], bounds[i + 1]
        k = best_prefix(pv[a:b][:max_k + 10], beta=beta, min_prob=min_prob, max_k=max_k)
        if k:
            out[s1[a]] = set(cid[a:a + k])
    return out


def tune(df, gt_d, s1_ids, beta=0.5, min_prob=0.02):
    """Grid-search the decoding rule on out-of-fold predictions. Returns (best, table)."""
    rows = []
    for exclusive in (False, True):
        for t in np.round(np.arange(0.30, 0.96, 0.05), 2):
            pred = decode(df, "threshold", threshold=t, exclusive=exclusive)
            rows.append({"mode": "threshold", "exclusive": exclusive, "threshold": float(t), "bias": 0.0,
                         "f05": macro_f05(pred, gt_d, s1_ids, beta)})
        for bias in (-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0):
            pred = decode(df, "expected_f", bias=bias, exclusive=exclusive, min_prob=min_prob, beta=beta)
            rows.append({"mode": "expected_f", "exclusive": exclusive, "threshold": None, "bias": bias,
                         "f05": macro_f05(pred, gt_d, s1_ids, beta)})
    table = pd.DataFrame(rows).sort_values("f05", ascending=False)
    return table.iloc[0].to_dict(), table
