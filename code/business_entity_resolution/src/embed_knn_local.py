"""Kaggle GPU job: multilingual-e5-small embeddings + exact top-K cosine neighbours (Step 5.5 / 10.1).

Input  (dataset amlc2026-text): train_text.parquet, test_text.parquet  [id, src, country, text]
Output (/kaggle/working):       knn_{split}.parquet  [s1 (u32), m (u32), src (u8), cos (f16)]
Per split and country: each S1 record queries the S2 pool and the S3 pool separately, keeping the top K each.
Model: intfloat/multilingual-e5-small (MIT, 118M params) - allowed by the challenge licence rule.
All work runs under the __main__ guard: the multi-GPU encode pool spawns processes that re-import this file.
"""
import glob
import os
import time

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

K = 6
BATCH = 128


def embed(model, texts, chunk=1_000_000):
    """Encode on all GPUs in 1M-text chunks, casting each chunk to float16 at once
    (a single fp32 pass over 12M texts needs ~18 GB and was OOM-killed). Returns L2-normalised float16."""
    t0 = time.time()
    pool = None
    if torch.cuda.device_count() > 1:
        pool = model.start_multi_process_pool([f"cuda:{i}" for i in range(torch.cuda.device_count())])
    out = []
    for i in range(0, len(texts), chunk):
        part = ["query: " + t for t in texts[i:i + chunk]]
        if pool is not None:
            e = model.encode_multi_process(part, pool, batch_size=BATCH, normalize_embeddings=True)
        else:
            e = model.encode(part, batch_size=BATCH, normalize_embeddings=True, convert_to_numpy=True)
        out.append(e.astype(np.float16))
        del e, part
        print(f"    encoded {min(i + chunk, len(texts)):,}/{len(texts):,} ({time.time()-t0:.0f}s)", flush=True)
    if pool is not None:
        model.stop_multi_process_pool(pool)
    return np.concatenate(out)


@torch.no_grad()
def topk(q: np.ndarray, p: np.ndarray, k: int, qb: int = 4096, pb: int = 50_000):
    """Exact inner-product top-k of q against p, tiled so it fits a 16 GB T4 (4096 x 500k fp16 block = 4 GB)."""
    P = [torch.from_numpy(p[i:i + pb]).cuda() for i in range(0, len(p), pb)]
    out_s, out_i = [], []
    for i in range(0, len(q), qb):
        Q = torch.from_numpy(q[i:i + qb]).cuda()
        best_s = best_i = None
        off = 0
        for blk in P:
            v, ix = (Q @ blk.T).topk(min(k, blk.shape[0]), dim=1)
            ix = ix + off
            off += blk.shape[0]
            if best_s is None:
                best_s, best_i = v, ix
            else:
                cs, ci = torch.cat([best_s, v], 1), torch.cat([best_i, ix], 1)
                best_s, j = cs.topk(k, dim=1)
                best_i = ci.gather(1, j)
        out_s.append(best_s.float().cpu().numpy())
        out_i.append(best_i.cpu().numpy())
    del P
    torch.cuda.empty_cache()
    return np.concatenate(out_s), np.concatenate(out_i)


def main():
    t0 = time.time()
    in_dir = "../../artifacts/kaggle_export"
    print("input dir", in_dir, "gpus", torch.cuda.device_count(), flush=True)
    model = SentenceTransformer("intfloat/multilingual-e5-small", device="cuda")
    model.max_seq_length = 32
    model.half()
    for split in ("test", "train"):
        df = pd.read_parquet(f"{in_dir}/{split}_text.parquet")
        print(split, len(df), f"({time.time()-t0:.0f}s)", flush=True)
        emb = embed(model, df["text"].tolist())
        print(f"  embedded {emb.shape} ({time.time()-t0:.0f}s)", flush=True)
        res = []
        for country in sorted(df["country"].unique()):
            s1 = np.where((df["country"] == country).values & (df["src"] == 1).values)[0]
            for src in (2, 3):
                pool = np.where((df["country"] == country).values & (df["src"] == src).values)[0]
                if len(s1) == 0 or len(pool) == 0:
                    continue
                s, ix = topk(emb[s1], emb[pool], K)
                res.append(pd.DataFrame({
                    "s1": np.repeat(df["id"].values[s1], K).astype(np.uint32),
                    "m": df["id"].values[pool][ix.ravel()].astype(np.uint32),
                    "src": np.uint8(src),
                    "cos": s.ravel().astype(np.float16),
                }))
                print(f"  {country} src{src}: {len(s1)} x {len(pool)} ({time.time()-t0:.0f}s)", flush=True)
        pd.concat(res).to_parquet(f"../../artifacts/kaggle_out/knn_{split}.parquet", compression="zstd")
        print(f"  wrote knn_{split}.parquet ({time.time()-t0:.0f}s)", flush=True)
        del emb, df, res
    print("DONE", f"{time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
