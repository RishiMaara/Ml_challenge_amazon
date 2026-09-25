"""Stage 2: pairwise features for every candidate pair (+ label on train).

Groups:
  * string similarity on names and addresses (rapidfuzz cpdist, C-speed, multithreaded)
  * token-set features weighted by IDF, incl. the IDF mass of *unmatched* tokens
    ("Star Coffee" vs "Star Coffee Roasters" -> 'roasters' is unmatched; key hard-negative signal)
  * structured address agreement: postcode, house number, numeric tokens, landmark, tail (city/state)
  * frequency features: how common the name is (chains/franchises need address evidence)
  * retrieval features carried over from blocking
  * relational features: how this pair ranks against the S1's other candidates and against
    the other S1 records competing for the same candidate
"""

import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler, Levenshtein

from .io_utils import load_ground_truth, save_report, Timer
from .normalize import token_idf
from .profile_data import normalized_records

try:
    import jellyfish
except ImportError:  # pragma: no cover
    jellyfish = None

LEGAL_TOKENS = {"pvt", "ltd", "corp", "inc", "co", "llc", "llp", "lp", "plc", "pllc", "sarl",
                "sas", "sa", "eurl", "sci", "snc", "gmbh", "ag", "bv", "nv", "opc"}

_G = {}


def _init(name_idf, addr_idf):
    _G["n"] = name_idf
    _G["a"] = addr_idf


def _idf_feats(ta, tb, idf, default):
    sa, sb = set(ta), set(tb)
    if not sa or not sb:
        return (np.nan,) * 5
    inter, union = sa & sb, sa | sb
    w = lambda s: sum(idf.get(t, default) for t in s)
    wi, wu, wa, wb = w(inter), w(union), w(sa), w(sb)
    un_a = [idf.get(t, default) for t in sa - sb]
    un_b = [idf.get(t, default) for t in sb - sa]
    return (wi / wu if wu else 0.0,
            wi / min(wa, wb) if min(wa, wb) else 0.0,
            sum(un_a) + sum(un_b),
            max(un_a + un_b) if (un_a or un_b) else 0.0,
            len(inter) / len(union))


def _char3(a, b):
    if not a or not b:
        return np.nan
    ga = {a[i:i + 3] for i in range(max(1, len(a) - 2))}
    gb = {b[i:i + 3] for i in range(max(1, len(b) - 2))}
    return len(ga & gb) / len(ga | gb)


def _eq3(a, b):
    """1 equal, 0 unknown (either missing), -1 conflict."""
    if not a or not b:
        return 0
    return 1 if a == b else -1


def _set_chunk(cols):
    (na, nb, nna, nnb, aa, ab, pa, pb, ha, hb, ma, mb, la, lb, ta, tb, ca, cb) = cols
    nidf, aidf = _G["n"], _G["a"]
    nd, ad = max(nidf.values(), default=10.0), max(aidf.values(), default=10.0)
    out = np.full((len(na), 27), np.nan, dtype=np.float32)
    for i in range(len(na)):
        tka, tkb = na[i].split(), nb[i].split()
        f = list(_idf_feats(tka, tkb, nidf, nd))
        sa, sb = set(tka), set(tkb)
        f.append(1.0 if (sa and sb and (sa <= sb or sb <= sa)) else 0.0)
        f.append(1.0 if (tka and tkb and tka[0] == tkb[0]) else 0.0)
        f.append(1.0 if (tka and tkb and tka[-1] == tkb[-1]) else 0.0)
        acr_a = "".join(t[0] for t in tka) if len(tka) >= 2 else ""
        acr_b = "".join(t[0] for t in tkb) if len(tkb) >= 2 else ""
        f.append(1.0 if ((acr_a and acr_a in sb) or (acr_b and acr_b in sa)) else 0.0)
        da = {t for t in tka if any(ch.isdigit() for ch in t)}
        db = {t for t in tkb if any(ch.isdigit() for ch in t)}
        f.append(0.0 if not (da or db) else (1.0 if da == db else -1.0))
        f.append(_char3(na[i], nb[i]))
        la_ = {t for t in nna[i].split() if t in LEGAL_TOKENS}
        lb_ = {t for t in nnb[i].split() if t in LEGAL_TOKENS}
        f.append(0.0 if not (la_ and lb_) else (1.0 if la_ & lb_ else -1.0))
        if jellyfish is not None and tka and tkb:
            f.append(1.0 if jellyfish.metaphone(tka[0]) == jellyfish.metaphone(tkb[0]) else 0.0)
        else:
            f.append(np.nan)
        # address
        f.extend(_idf_feats(aa[i].split(), ab[i].split(), aidf, ad))
        f.append(_eq3(pa[i], pb[i]))
        f.append(1.0 if (pa[i] and pb[i] and pa[i][:3] == pb[i][:3]) else 0.0)
        f.append(_eq3(ha[i], hb[i]))
        sna, snb = set(ma[i].split()), set(mb[i].split())
        f.append(len(sna & snb) / len(sna | snb) if (sna and snb) else np.nan)
        f.append(float(len(sna ^ snb)) if (sna and snb) else np.nan)
        sla, slb = set(la[i].split()), set(lb[i].split())
        f.append(len(sla & slb) / len(sla | slb) if (sla and slb) else np.nan)
        sta, stb = set(ta[i].split()), set(tb[i].split())
        f.append(len(sta & stb) / len(sta | stb) if (sta and stb) else np.nan)
        # cross-field: name tokens found in the other record's address (DBA / name-in-address)
        aa_set, ab_set = set(aa[i].split()), set(ab[i].split())
        x1 = len(sa & ab_set) / len(sa) if sa else np.nan
        x2 = len(sb & aa_set) / len(sb) if sb else np.nan
        f.append(np.nanmax([x1, x2]) if not (np.isnan(x1) and np.isnan(x2)) else np.nan)
        f.append(1.0 if ca[i] == cb[i] else 0.0)
        out[i] = f
    return out


SET_COLS = ["n_idf_jacc", "n_idf_mincov", "n_unmatched_idf", "n_unmatched_idf_max", "n_jacc",
            "n_contain", "n_first_eq", "n_last_eq", "n_acronym", "n_digit_eq", "n_char3_jacc",
            "n_legal_eq", "n_phon_first_eq",
            "a_idf_jacc", "a_idf_mincov", "a_unmatched_idf", "a_unmatched_idf_max", "a_jacc",
            "pc_eq", "pc_prefix3_eq", "house_eq", "nums_jacc", "nums_symdiff",
            "landmark_jacc", "tail_jacc", "name_in_addr", "country_eq"]


def _string_feats(a, b, prefix, workers):
    f = {}
    f[f"{prefix}_ratio"] = process.cpdist(a, b, scorer=fuzz.ratio, workers=workers)
    f[f"{prefix}_partial"] = process.cpdist(a, b, scorer=fuzz.partial_ratio, workers=workers)
    f[f"{prefix}_tsort"] = process.cpdist(a, b, scorer=fuzz.token_sort_ratio, workers=workers)
    f[f"{prefix}_tset"] = process.cpdist(a, b, scorer=fuzz.token_set_ratio, workers=workers)
    f[f"{prefix}_jw"] = process.cpdist(a, b, scorer=JaroWinkler.normalized_similarity, workers=workers)
    f[f"{prefix}_lev"] = process.cpdist(a, b, scorer=Levenshtein.distance, workers=workers)
    f[f"{prefix}_levn"] = process.cpdist(a, b, scorer=Levenshtein.normalized_similarity, workers=workers)
    return {k: np.asarray(v, dtype=np.float32) for k, v in f.items()}


def _margin(v, key, g):
    """v minus the best competing value in the same group (second best if v is the best)."""
    top = g.transform("max")
    r = g.rank(ascending=False, method="first")
    second = v.where(r == 2).groupby(key, sort=False).transform("max")
    return (v - np.where(r == 1, second, top)).astype(np.float32)


def relational(df, cols):
    """Rank/margin of each pair versus competitors on both sides of the match."""
    k1 = [df["s1_id"], df["tgt_source"]]
    k2 = df["cand_id"]
    out = {"n_cands_s1": df.groupby(k1, sort=False)["cand_id"].transform("size").astype(np.float32),
           "n_s1_for_cand": df.groupby(k2, sort=False)["s1_id"].transform("size").astype(np.float32)}
    for c in cols:
        v = df[c].astype(np.float32)
        g1 = v.groupby(k1, sort=False)
        g2 = v.groupby(k2, sort=False)
        out[f"{c}_rk_s1"] = g1.rank(ascending=False, method="min").astype(np.float32)
        out[f"{c}_margin_s1"] = _margin(v, k1, g1)
        out[f"{c}_rk_cand"] = g2.rank(ascending=False, method="min").astype(np.float32)
        out[f"{c}_margin_cand"] = _margin(v, k2, g2)
    return pd.DataFrame(out, index=df.index)


def build(cfg, split):
    recs = normalized_records(cfg, split).drop_duplicates("entity_id").set_index("entity_id")
    cands = pd.read_parquet(os.path.join(cfg.work_dir, f"cands_{split}.parquet"))
    print(f"{split}: {len(cands):,} candidate pairs")

    with Timer("frequency features"):
        s1 = recs[recs["source"] == 1]
        freq_s1 = s1.groupby(["country_key", "name_core"]).size()
        freq_all = recs.groupby(["country_key", "name_core"]).size()
        pc_cnt = recs[recs["postcode"] != ""].groupby(["country_key", "postcode"]).size()

    fields = ["name_core", "name_norm", "addr_norm", "postcode", "house_no", "addr_nums",
              "landmark", "addr_tail", "country_key"]
    A = recs.reindex(cands["s1_id"].values)[fields].reset_index(drop=True)
    B = recs.reindex(cands["cand_id"].values)[fields].reset_index(drop=True)
    A = A.fillna(""); B = B.fillna("")

    feats = {}
    with Timer("string similarity (rapidfuzz cpdist)"):
        feats.update(_string_feats(A["name_core"].tolist(), B["name_core"].tolist(), "n", cfg.n_jobs))
        feats.update(_string_feats(A["addr_norm"].tolist(), B["addr_norm"].tolist(), "a", cfg.n_jobs))
        feats["nn_tset"] = np.asarray(process.cpdist(A["name_norm"].tolist(), B["name_norm"].tolist(),
                                                     scorer=fuzz.token_set_ratio, workers=cfg.n_jobs), np.float32)
        full_a = (A["name_core"] + " " + A["addr_norm"]).tolist()
        full_b = (B["name_core"] + " " + B["addr_norm"]).tolist()
        feats["full_tset"] = np.asarray(process.cpdist(full_a, full_b, scorer=fuzz.token_set_ratio,
                                                       workers=cfg.n_jobs), np.float32)

    with Timer("token/IDF/structured features (multiprocess)"):
        name_idf, _ = token_idf([recs["name_core"]])
        addr_idf, _ = token_idf([recs["addr_norm"]])
        order = ["name_core", "name_core", "name_norm", "name_norm", "addr_norm", "addr_norm",
                 "postcode", "postcode", "house_no", "house_no", "addr_nums", "addr_nums",
                 "landmark", "landmark", "addr_tail", "addr_tail", "country_key", "country_key"]
        n = len(cands)
        jobs = []
        for s in range(0, n, cfg.feature_chunk):
            e = min(n, s + cfg.feature_chunk)
            cols = []
            for j, fld in enumerate(order):
                src = A if j % 2 == 0 else B
                cols.append(src[fld].values[s:e].tolist())
            jobs.append(cols)
        with ProcessPoolExecutor(max_workers=cfg.n_jobs, initializer=_init,
                                 initargs=(name_idf, addr_idf)) as ex:
            parts = list(ex.map(_set_chunk, jobs))
        setf = np.vstack(parts) if parts else np.empty((0, len(SET_COLS)), np.float32)
        for j, c in enumerate(SET_COLS):
            feats[c] = setf[:, j]

    F = pd.DataFrame(feats)
    F["n_empty"] = ((A["name_core"] == "") | (B["name_core"] == "")).astype(np.int8).values
    F["a_empty_s1"] = (A["addr_norm"] == "").astype(np.int8).values
    F["a_empty_cand"] = (B["addr_norm"] == "").astype(np.int8).values
    F["n_len_s1"] = A["name_core"].str.split().str.len().astype(np.float32).values
    F["n_len_cand"] = B["name_core"].str.split().str.len().astype(np.float32).values
    F["a_len_s1"] = A["addr_norm"].str.len().astype(np.float32).values
    F["a_len_cand"] = B["addr_norm"].str.len().astype(np.float32).values
    F["pc_both"] = ((A["postcode"] != "") & (B["postcode"] != "")).astype(np.int8).values
    key_a = list(zip(A["country_key"], A["name_core"]))
    F["name_freq_s1"] = freq_s1.reindex(key_a).fillna(0).astype(np.float32).values
    F["name_freq_all"] = freq_all.reindex(key_a).fillna(0).astype(np.float32).values
    F["pc_count"] = pc_cnt.reindex(list(zip(A["country_key"], A["postcode"]))).fillna(0).astype(np.float32).values

    out = pd.concat([cands.reset_index(drop=True), F], axis=1)
    out["pair_sim"] = (0.6 * out["n_idf_jacc"].fillna(0) + 0.4 * out["a_idf_jacc"].fillna(0)).astype(np.float32)
    with Timer("relational features"):
        rel = relational(out, ["pair_sim", "n_tset", "a_tset", "n_idf_jacc", "rrf"])
        out = pd.concat([out, rel], axis=1)

    if split == "train":
        gt, _ = load_ground_truth(cfg)
        out = out.merge(gt.assign(y=np.int8(1)), on=["s1_id", "cand_id"], how="left")
        out["y"] = out["y"].fillna(0).astype(np.int8)
    path = os.path.join(cfg.work_dir, f"feats_{split}.parquet")
    out.to_parquet(path, index=False)

    rep = {"split": split, "pairs": int(len(out)), "n_features": int(len(feature_columns(out)))}
    if split == "train":
        rep["positives"] = int(out["y"].sum())
        rep["pos_rate"] = round(float(out["y"].mean()), 5)
        pos, neg = out[out["y"] == 1], out[out["y"] == 0]
        rep["feature_medians_pos_vs_neg"] = {
            c: [round(float(pos[c].median()), 3), round(float(neg[c].median()), 3)]
            for c in ["n_tset", "n_idf_jacc", "n_unmatched_idf", "a_tset", "a_idf_jacc", "pc_eq",
                      "house_eq", "rrf", "pair_sim_rk_s1", "pair_sim_rk_cand", "name_freq_s1"]}
    save_report(cfg, f"02_features_{split}", rep)


NON_FEATURES = {"s1_id", "cand_id", "y"}


def feature_columns(df):
    return [c for c in df.columns if c not in NON_FEATURES]
