"""Matcher training (Step 8) + first decision layer.

    python -m src.train --fit ../../artifacts/features/fit_sample --valid ../../artifacts/features/dev \
        --model-out ../../artifacts/models/lgb_v1.txt

Decision (v1): p(pair) -> keep each record only for its highest-p S1 (one-owner constraint, EDA 2.4)
-> keep pairs with p >= threshold (threshold tuned for macro F0.5 on the validation features).
"""
import argparse
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

from .evaluate import evaluate
from .features import BLOCK_COLS, CTX_COLS

ID_COLS = ["s1", "m", "label"]
PARAMS = dict(objective="binary", learning_rate=0.08, num_leaves=127, min_data_in_leaf=200, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, max_bin=255, num_threads=0, verbose=-1, seed=2026)


def load_feats(d: Path) -> pl.DataFrame:
    return pl.read_parquet(Path(d) / "*.parquet")


def feature_names(df: pl.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in ID_COLS]


def to_xy(df: pl.DataFrame, feats: list[str]):
    X = df.select([pl.col(c).cast(pl.Float32) for c in feats]).to_numpy()
    y = df["label"].to_numpy() if "label" in df.columns else None
    return X, y


def one_owner(pred: pl.DataFrame, p_col: str = "p") -> pl.DataFrame:
    """Keep each (m, src) only for its best S1 (ties: keep all tied)."""
    return pred.filter(pl.col(p_col) >= pl.col(p_col).max().over("m", "src"))


def to_pairs(pred: pl.DataFrame) -> pl.DataFrame:
    return pred.select((pl.lit("S1-") + pl.col("s1").cast(pl.Utf8)).alias("s1_id"),
                       (pl.lit("S") + pl.col("src").cast(pl.Utf8) + "-" + pl.col("m").cast(pl.Utf8)).alias("match_id"))


def tune_threshold(pred: pl.DataFrame, truth: pl.DataFrame, s1_ids: pl.DataFrame, grid=None) -> tuple[float, list]:
    grid = grid if grid is not None else np.round(np.arange(0.20, 0.91, 0.05), 2)
    owned = one_owner(pred)
    rows = []
    for t in grid:
        r = evaluate(to_pairs(owned.filter(pl.col("p") >= t)), truth, s1_ids)
        rows.append((float(t), r["f05"], r["macro_p"], r["macro_r"]))
    best = max(rows, key=lambda x: x[1])
    return best[0], rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", type=Path, required=True)
    ap.add_argument("--valid", type=Path, required=True)
    ap.add_argument("--model-out", type=Path, required=True)
    ap.add_argument("--rounds", type=int, default=2000)
    ap.add_argument("--gt", type=Path, default=Path("../../artifacts/processed/train_gt_pairs.parquet"))
    ap.add_argument("--split-file", type=Path, default=Path("../../artifacts/splits/split_v1.parquet"))
    ap.add_argument("--drop", nargs="*", default=[], help="feature columns to exclude (ablation)")
    args = ap.parse_args()
    t0 = time.time()
    fit, val = load_feats(args.fit), load_feats(args.valid)
    feats = [f for f in feature_names(fit) if f not in args.drop]
    Xf, yf = to_xy(fit, feats)
    Xv, yv = to_xy(val, feats)
    print(f"fit {Xf.shape}, pos {yf.mean():.4f} | valid {Xv.shape}, pos {yv.mean():.4f} | {len(feats)} features ({time.time()-t0:.0f}s)")
    dfit = lgb.Dataset(Xf, yf, feature_name=feats, free_raw_data=True)
    dval = lgb.Dataset(Xv, yv, reference=dfit)
    booster = lgb.train(PARAMS, dfit, args.rounds, valid_sets=[dval], valid_names=["valid"],
                        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)])
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(args.model_out))
    print(f"trained {booster.best_iteration} rounds ({time.time()-t0:.0f}s)")

    val = val.select("s1", "m", "src", "label").with_columns(pl.Series("p", booster.predict(Xv, num_iteration=booster.best_iteration)))
    sp = pl.read_parquet(args.split_file)
    s1_ids = sp.filter(pl.col("s1_id").str.slice(3).cast(pl.UInt32).is_in(val["s1"].unique().implode())).select("s1_id", "country")
    # every S1 of the eval subset is scored, including those with no candidates
    subset = sp.filter(pl.col("is_dev")) if s1_ids.height <= 25_000 else sp.filter(pl.col("role") == "val")
    truth = pl.read_parquet(args.gt)
    thr, curve = tune_threshold(val, truth, subset.select("s1_id", "country"))
    for t, f, p, r in curve:
        print(f"  thr {t:.2f}: F0.5 {f:.4f}  P {p:.4f}  R {r:.4f}")
    final = evaluate(to_pairs(one_owner(val).filter(pl.col("p") >= thr)), truth, subset.select("s1_id", "country"))
    print("BEST thr", thr, {k: round(v, 4) for k, v in final.items() if isinstance(v, float)})
    for c, v in final["by_country"].items():
        print("   ", c, {k: round(x, 4) for k, x in v.items() if isinstance(x, float)})
    imp = sorted(zip(feats, booster.feature_importance("gain")), key=lambda x: -x[1])
    print("top gain features:", [(f, round(g / sum(x[1] for x in imp), 3)) for f, g in imp[:20]])
    meta = {"threshold": thr, "features": feats, "best_iteration": booster.best_iteration, "params": PARAMS,
            "valid_f05": final["f05"]}
    args.model_out.with_suffix(".json").write_text(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
