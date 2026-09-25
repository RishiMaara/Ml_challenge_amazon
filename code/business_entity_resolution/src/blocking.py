"""Stage 1: candidate generation (blocking).

Several complementary retrievers run inside each country partition and separately for
each target source (S2, S3), so a crowded S2 never pushes S3 matches out of the top-k:

  name_char : char 3-4gram TF-IDF on name_core      (typos, spacing, transliteration)
  name_word : word 1-2gram TF-IDF on name_core      (word reordering, abbreviation)
  full_word : word TF-IDF on name_core + address    (BM25-like; rescues renamed/DBA records)
  addr_char : char 3-4gram TF-IDF on address        (same address, different name form)
  key_*     : exact keys - name_core, postcode+first name token, acronym+postcode

Hits are merged, scored with reciprocal-rank fusion (RRF) and capped at
`cap_per_source` per (S1, source). All retriever scores/ranks are kept as model features.
TF-IDF vocabularies are fit on the split's own records (no labels), so test-only
countries get their own statistics.
"""

import os

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from .io_utils import load_ground_truth, save_report, Timer
from .profile_data import normalized_records

try:
    from sparse_dot_topn import sp_matmul_topn
except ImportError:  # pragma: no cover
    sp_matmul_topn = None

RETRIEVERS = ["name_char", "name_word", "full_word", "addr_char"]
KEYS = ["key_name", "key_pc_tok", "key_acr_pc"]
RRF_K = 10.0


def _vectorizer(kind):
    common = dict(sublinear_tf=True, dtype=np.float32, lowercase=False)
    if kind in ("name_char", "addr_char"):
        return TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4), min_df=2, max_df=0.5, **common)
    if kind == "name_word":
        return TfidfVectorizer(analyzer="word", token_pattern=r"\S+", ngram_range=(1, 2),
                               min_df=1, max_df=0.3, **common)
    return TfidfVectorizer(analyzer="word", token_pattern=r"\S+", ngram_range=(1, 1),
                           min_df=1, max_df=0.3, **common)


def _texts(part, kind):
    if kind in ("name_char", "name_word"):
        return part["name_core"].tolist()
    if kind == "addr_char":
        return part["addr_norm"].tolist()
    return (part["name_core"] + " " + part["addr_norm"]).tolist()


def topk(A, BT, k, threshold, n_threads):
    """Top-k cosine neighbours of each row of A among columns of BT (rows L2-normalised)."""
    if A.shape[0] == 0 or BT.shape[1] == 0:
        return np.empty(0, np.int64), np.empty(0, np.int64), np.empty(0, np.float32)
    if sp_matmul_topn is not None:
        C = sp_matmul_topn(A, BT, top_n=k, threshold=threshold, sort=True,
                           n_threads=n_threads if n_threads > 1 else None)
        C = C.tocoo()
        return C.row.astype(np.int64), C.col.astype(np.int64), C.data.astype(np.float32)
    C = (A @ BT).tocsr()  # fallback: slower, more memory
    rows, cols, vals = [], [], []
    for i in range(C.shape[0]):
        s, e = C.indptr[i], C.indptr[i + 1]
        d, c = C.data[s:e], C.indices[s:e]
        keep = d > threshold
        d, c = d[keep], c[keep]
        if len(d) > k:
            idx = np.argpartition(-d, k)[:k]
            d, c = d[idx], c[idx]
        rows.append(np.full(len(d), i)); cols.append(c); vals.append(d)
    return np.concatenate(rows), np.concatenate(cols), np.concatenate(vals).astype(np.float32)


def _key_index(values, idx, max_block):
    """key -> array of target positions, skipping empty keys and oversized blocks."""
    s = pd.Series(idx, index=values)
    s = s[s.index != ""]
    groups = s.groupby(level=0).agg(list)
    return {k: v for k, v in groups.items() if len(v) <= max_block}


def _keys(part):
    first = part["name_core"].str.split(" ").str[0].fillna("")
    pc = part["postcode"]
    return {
        "key_name": part["name_core"].where(part["name_core"].str.len() >= 3, "").values,
        "key_pc_tok": np.where((pc != "") & (first.str.len() >= 2), pc + "|" + first, ""),
        "key_acr_pc": np.where((pc != "") & (part["acronym"] != ""), pc + "|" + part["acronym"], ""),
    }


def block_partition(cfg, part, writer_rows):
    s1 = part[part["source"] == 1].reset_index(drop=True)
    if len(s1) == 0:
        return
    tgt = {t: part[part["source"] == t].reset_index(drop=True) for t in (2, 3)}
    k_of = {"name_char": cfg.k_name_char, "name_word": cfg.k_name_word,
            "full_word": cfg.k_full_word, "addr_char": cfg.k_addr_char}

    mats = {}
    for kind in RETRIEVERS:
        vec = _vectorizer(kind)
        try:
            vec.fit(_texts(part, kind))
        except ValueError:  # empty vocabulary (e.g. no addresses in this partition)
            continue
        A = vec.transform(_texts(s1, kind)).tocsr()
        mats[kind] = (A, {t: vec.transform(_texts(tgt[t], kind)).T.tocsr() for t in (2, 3)})

    s1_keys = _keys(s1)
    key_idx = {}
    for t in (2, 3):
        tk = _keys(tgt[t])
        key_idx[t] = {k: _key_index(tk[k], np.arange(len(tgt[t])), cfg.max_key_block) for k in KEYS}

    for start in range(0, len(s1), cfg.chunk_rows):
        stop = min(start + cfg.chunk_rows, len(s1))
        for t in (2, 3):
            if len(tgt[t]) == 0:
                continue
            frames = []
            for kind, (A, BTs) in mats.items():
                r, c, v = topk(A[start:stop], BTs[t], k_of[kind], cfg.min_score, cfg.n_jobs)
                f = pd.DataFrame({"r": r + start, "c": c, "score": v})
                f["rank"] = f.groupby("r")["score"].rank(ascending=False, method="first").astype(np.float32)
                f["ret"] = kind
                frames.append(f)
            for k in KEYS:
                idx = key_idx[t][k]
                rs, cs = [], []
                for i in range(start, stop):
                    hit = idx.get(s1_keys[k][i])
                    if hit is not None:
                        rs.extend([i] * len(hit)); cs.extend(hit)
                if rs:
                    frames.append(pd.DataFrame({"r": rs, "c": cs, "score": 1.0, "rank": 1.0, "ret": k}))
            if not frames:
                continue
            long = pd.concat(frames, ignore_index=True)
            wide = long.pivot_table(index=["r", "c"], columns="ret", values=["score", "rank"], aggfunc="max")
            wide.columns = [f"{'sc' if a == 'score' else 'rk'}_{b}" for a, b in wide.columns]
            wide = wide.reset_index()
            for kind in RETRIEVERS:
                for p in ("sc", "rk"):
                    if f"{p}_{kind}" not in wide:
                        wide[f"{p}_{kind}"] = np.nan
            for k in KEYS:
                wide[k] = wide[f"sc_{k}"].fillna(0).astype(np.int8) if f"sc_{k}" in wide else np.int8(0)
            wide = wide.drop(columns=[c for c in wide.columns if c.startswith(("sc_key", "rk_key"))])
            rrf = np.zeros(len(wide), dtype=np.float32)
            for kind in RETRIEVERS:
                rrf += np.nan_to_num(1.0 / (RRF_K + wide[f"rk_{kind}"].values), nan=0.0)
            for k in KEYS:
                rrf += wide[k].values * (1.0 / (RRF_K + 1))
            wide["rrf"] = rrf
            wide["n_ret"] = wide[[f"sc_{k}" for k in RETRIEVERS]].notna().sum(axis=1).astype(np.int8) \
                + wide[KEYS].sum(axis=1).astype(np.int8)
            wide["rrf_rank"] = wide.groupby("r")["rrf"].rank(ascending=False, method="first").astype(np.int16)
            wide["s1_id"] = s1["entity_id"].values[wide["r"].values]
            wide["cand_id"] = tgt[t]["entity_id"].values[wide["c"].values]
            wide["tgt_source"] = np.int8(t)
            wide = wide.drop(columns=["r", "c"])
            writer_rows.append(wide)


def run(cfg, split):
    df = normalized_records(cfg, split).drop_duplicates("entity_id")
    out_path = os.path.join(cfg.work_dir, f"cands_full_{split}.parquet")
    writer, schema = None, None
    total = 0
    with Timer(f"blocking {split}"):
        for country, part in df.groupby("country_key", sort=True):
            rows = []
            with Timer(f"  partition {country!r}: {len(part):,} records"):
                block_partition(cfg, part.reset_index(drop=True), rows)
            if not rows:
                continue
            tab = pd.concat(rows, ignore_index=True)
            cols = ["s1_id", "cand_id", "tgt_source", "rrf", "rrf_rank", "n_ret"] + \
                [f"{p}_{k}" for k in RETRIEVERS for p in ("sc", "rk")] + KEYS
            tab = tab[cols]
            for c in tab.columns:
                if tab[c].dtype == np.float64:
                    tab[c] = tab[c].astype(np.float32)
            at = pa.Table.from_pandas(tab, preserve_index=False)
            if writer is None:
                schema = at.schema
                writer = pq.ParquetWriter(out_path, schema)
            writer.write_table(at.cast(schema))
            total += len(tab)
    if writer is not None:
        writer.close()
    print(f"{split}: {total:,} raw candidate pairs -> {out_path}")

    full = pd.read_parquet(out_path)
    capped = full[full["rrf_rank"] <= cfg.cap_per_source].reset_index(drop=True)
    capped.to_parquet(os.path.join(cfg.work_dir, f"cands_{split}.parquet"), index=False)

    rep = {"split": split, "raw_pairs": int(len(full)), "capped_pairs": int(len(capped)),
           "cap_per_source": cfg.cap_per_source}
    n1 = int((df["source"] == 1).sum())
    n23 = int((df["source"] != 1).sum())
    rep["s1_records"] = n1
    rep["avg_candidates_per_s1"] = round(len(capped) / max(1, n1), 2)
    rep["reduction_ratio"] = round(1 - len(capped) / max(1, n1 * n23), 7)
    if split == "train":
        rep.update(recall_report(cfg, df, full))
    save_report(cfg, f"01_blocking_{split}", rep)


def _macro_f05_ceiling(gt_d, s1_ids, found):
    """Macro F0.5 of a perfect matcher restricted to the candidate set (singletons -> 1.0)."""
    tot = 0.0
    for s in s1_ids:
        g = gt_d.get(s)
        if not g:
            tot += 1.0
            continue
        tp = len(found.get(s, ()))
        if tp:
            p, r = 1.0, tp / len(g)
            tot += 1.25 * p * r / (0.25 * p + r)
    return tot / max(1, len(s1_ids))


def recall_report(cfg, df, full):
    gt, _ = load_ground_truth(cfg)
    gt = gt.assign(tgt_source=np.where(gt["cand_id"].str.startswith("S2-"), 2, 3))
    gt_d = {}
    for s, c in zip(gt["s1_id"], gt["cand_id"]):
        gt_d.setdefault(s, set()).add(c)
    s1_ids = df.loc[df["source"] == 1, "entity_id"].tolist()
    pos = full.merge(gt[["s1_id", "cand_id"]], on=["s1_id", "cand_id"], how="inner")
    ranks = full["rrf_rank"].values
    n_true = len(gt)
    rep = {"true_pairs": n_true}
    rep["recall_uncapped"] = round(len(pos) / max(1, n_true), 5)
    rec_cap = {}
    for cap in (1, 2, 3, 5, 10, 15, 20, 25, 30, 40, 60):
        sel = pos[pos["rrf_rank"] <= cap]
        n_pairs = int((ranks <= cap).sum())
        rec_cap[cap] = {"recall": round(len(sel) / max(1, n_true), 5),
                        "pairs_per_s1": round(n_pairs / max(1, len(s1_ids)), 2)}
    rep["recall_at_cap"] = rec_cap
    for t in (2, 3):
        n_t = int((gt["tgt_source"] == t).sum())
        sel = pos[(pos["tgt_source"] == t) & (pos["rrf_rank"] <= cfg.cap_per_source)]
        rep[f"recall_capped_source{t}"] = round(len(sel) / max(1, n_t), 5)
    per_ret, only = {}, {}
    hit_cols = {k: pos[f"sc_{k}"].notna() for k in RETRIEVERS}
    hit_cols.update({k: pos[k] == 1 for k in KEYS})
    any_other = {}
    for k, h in hit_cols.items():
        per_ret[k] = round(h.sum() / max(1, n_true), 5)
        others = np.zeros(len(pos), bool)
        for k2, h2 in hit_cols.items():
            if k2 != k:
                others |= h2.values
        only[k] = int((h.values & ~others).sum())
    rep["recall_per_retriever"] = per_ret
    rep["true_pairs_found_only_by"] = only
    capped_pos = pos[pos["rrf_rank"] <= cfg.cap_per_source]
    found = capped_pos.groupby("s1_id")["cand_id"].agg(set).to_dict()
    rep["macro_f05_ceiling_capped"] = round(_macro_f05_ceiling(gt_d, s1_ids, found), 5)
    cc = df.set_index("entity_id")["country_key"]
    missed = gt.merge(capped_pos[["s1_id", "cand_id"]].assign(f=1), on=["s1_id", "cand_id"], how="left")
    missed = missed[missed["f"].isna()]
    missed = missed.assign(country=cc.reindex(missed["s1_id"]).values,
                           cand_country=cc.reindex(missed["cand_id"]).values)
    rep["missed_true_pairs"] = int(len(missed))
    rep["missed_cross_country"] = int((missed["country"] != missed["cand_country"]).sum())
    rep["missed_by_country"] = missed["country"].value_counts().to_dict()
    recs = df.set_index("entity_id")
    smp = missed.sample(min(25, len(missed)), random_state=cfg.seed)
    rep["missed_samples"] = [
        {"s1": f"{recs.at[a, 'business_name']} | {recs.at[a, 'business_address']} | {recs.at[a, 'country']}",
         "true": (f"{recs.at[b, 'business_name']} | {recs.at[b, 'business_address']} | {recs.at[b, 'country']}"
                  if b in recs.index else "<id not in sources>")}
        for a, b in zip(smp["s1_id"], smp["cand_id"])]
    return rep
