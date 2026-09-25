"""Stage 3: train the matcher with GroupKFold over S1 entities, stack, tune decoding.

Base models : LightGBM, XGBoost, CatBoost (each 5-fold, out-of-fold predictions kept)
Stacker     : logistic regression on base-model logits (fit on OOF)
Decoding    : grid over threshold / expected-F0.5 rule, exclusivity on/off, scored with the
              exact macro F0.5 over *all* train S1 entities (blocking misses count as errors)
Reports     : pair-level AUC/AP per model, macro F0.5 per country, leave-one-country-out
              score (proxy for the unseen test country), top features, FP/FN samples.
"""

import json
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold

from .decode import decode, macro_f05, tune
from .features import feature_columns
from .io_utils import load_ground_truth, gt_dict, save_report, Timer
from .profile_data import normalized_records


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def fit_lgb(cfg, Xtr, ytr, Xva, yva):
    import lightgbm as lgb
    params = dict(objective="binary", learning_rate=cfg.lgb_lr, num_leaves=cfg.lgb_leaves,
                  min_data_in_leaf=100, feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1,
                  lambda_l2=1.0, max_bin=255, num_threads=cfg.n_jobs, verbose=-1, seed=cfg.seed)
    dtr = lgb.Dataset(Xtr, ytr, free_raw_data=True)
    dva = lgb.Dataset(Xva, yva, reference=dtr)
    m = lgb.train(params, dtr, cfg.lgb_rounds, valid_sets=[dva],
                  callbacks=[lgb.early_stopping(cfg.early_stop, verbose=False), lgb.log_evaluation(0)])
    return m, m.predict(Xva, num_iteration=m.best_iteration)


def fit_xgb(cfg, Xtr, ytr, Xva, yva):
    import xgboost as xgb
    m = xgb.XGBClassifier(n_estimators=cfg.lgb_rounds, learning_rate=0.05, max_depth=8,
                          min_child_weight=5, subsample=0.8, colsample_bytree=0.7, reg_lambda=1.0,
                          tree_method="hist", n_jobs=cfg.n_jobs, eval_metric="logloss",
                          early_stopping_rounds=cfg.early_stop, random_state=cfg.seed)
    m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    return m, m.predict_proba(Xva)[:, 1]


def fit_cat(cfg, Xtr, ytr, Xva, yva):
    from catboost import CatBoostClassifier
    m = CatBoostClassifier(iterations=cfg.lgb_rounds, learning_rate=0.08, depth=8,
                           l2_leaf_reg=3, thread_count=cfg.n_jobs, random_seed=cfg.seed,
                           od_type="Iter", od_wait=cfg.early_stop, verbose=False)
    m.fit(Xtr, ytr, eval_set=(Xva, yva))
    return m, m.predict_proba(Xva)[:, 1]


def predict_model(kind, m, X):
    if kind == "lgb":
        return m.predict(X, num_iteration=m.best_iteration)
    return m.predict_proba(X)[:, 1]


FITTERS = {"lgb": fit_lgb, "xgb": fit_xgb, "cat": fit_cat}


def run(cfg, loco=True):
    df = pd.read_parquet(os.path.join(cfg.work_dir, "feats_train.parquet"))
    feats = feature_columns(df)
    X = df[feats].astype(np.float32).values
    y = df["y"].values.astype(np.int8)
    groups = df["s1_id"].values
    kinds = ["lgb"] + (["xgb"] if cfg.use_xgboost else []) + (["cat"] if cfg.use_catboost else [])
    print(f"train pairs {len(df):,}, positives {int(y.sum()):,}, features {len(feats)}, models {kinds}")

    folds = list(GroupKFold(n_splits=cfg.n_folds).split(X, y, groups))
    oof = {k: np.zeros(len(df), dtype=np.float32) for k in kinds}
    models = {k: [] for k in kinds}
    rep = {"pairs": int(len(df)), "positives": int(y.sum()), "features": len(feats), "models": {}}
    for k in kinds:
        with Timer(f"{k} {cfg.n_folds}-fold"):
            for i, (tr, va) in enumerate(folds):
                m, p = FITTERS[k](cfg, X[tr], y[tr], X[va], y[va])
                oof[k][va] = p
                models[k].append(m)
        rep["models"][k] = {"auc": round(roc_auc_score(y, oof[k]), 6),
                            "ap": round(average_precision_score(y, oof[k]), 6),
                            "logloss": round(log_loss(y, np.clip(oof[k], 1e-6, 1 - 1e-6)), 6)}
        print(k, rep["models"][k])

    # stacker on logits, evaluated out-of-fold with the same folds
    Z = np.column_stack([_logit(oof[k]) for k in kinds])
    stack_oof = np.zeros(len(df), dtype=np.float32)
    for tr, va in folds:
        lr = LogisticRegression(C=1.0, max_iter=1000).fit(Z[tr], y[tr])
        stack_oof[va] = lr.predict_proba(Z[va])[:, 1]
    stacker = LogisticRegression(C=1.0, max_iter=1000).fit(Z, y)
    rep["models"]["stack"] = {"auc": round(roc_auc_score(y, stack_oof), 6),
                              "ap": round(average_precision_score(y, stack_oof), 6),
                              "logloss": round(log_loss(y, np.clip(stack_oof, 1e-6, 1 - 1e-6)), 6),
                              "weights": dict(zip(kinds, np.round(stacker.coef_[0], 4).tolist()))}

    # decoding on the exact challenge metric over ALL train S1 entities
    recs = normalized_records(cfg, "train").drop_duplicates("entity_id")
    s1_ids = recs.loc[recs["source"] == 1, "entity_id"].tolist()
    gt, _ = load_ground_truth(cfg)
    gt_d = gt_dict(gt)
    best_name, best_cfg, best_f = None, None, -1
    tables = {}
    for name, p in list(oof.items()) + [("stack", stack_oof)]:
        pdf = pd.DataFrame({"s1_id": df["s1_id"].values, "cand_id": df["cand_id"].values, "p": p})
        b, table = tune(pdf, gt_d, s1_ids, beta=cfg.beta, min_prob=cfg.min_prob)
        tables[name] = table.head(8).to_dict("records")
        rep["models"][name]["best_macro_f05"] = round(b["f05"], 5)
        if b["f05"] > best_f:
            best_name, best_cfg, best_f = name, b, b["f05"]
    rep["decode_top_configs"] = tables
    rep["chosen"] = {"model": best_name, **{k: v for k, v in best_cfg.items()}}

    final_p = stack_oof if best_name == "stack" else oof[best_name]
    pdf = pd.DataFrame({"s1_id": df["s1_id"].values, "cand_id": df["cand_id"].values, "p": final_p})
    pred = decode(pdf, best_cfg["mode"], threshold=best_cfg["threshold"] or 0.5, bias=best_cfg["bias"],
                  exclusive=best_cfg["exclusive"], min_prob=cfg.min_prob, beta=cfg.beta)
    f, per = macro_f05(pred, gt_d, s1_ids, cfg.beta, per_entity=True)
    rep.update(breakdown(recs, s1_ids, per, pred, gt_d))
    rep.update(error_samples(recs, pdf, pred, gt_d))

    imp = pd.Series(models["lgb"][0].feature_importance("gain"), index=feats)
    for m in models["lgb"][1:]:
        imp += pd.Series(m.feature_importance("gain"), index=feats)
    imp = (imp / imp.sum()).sort_values(ascending=False)
    rep["top_features_gain_pct"] = {k: round(100 * v, 2) for k, v in imp.head(35).items()}
    rep["zero_gain_features"] = imp[imp <= 0].index.tolist()

    if loco:
        rep["leave_one_country_out"] = leave_one_country_out(cfg, df, X, y, recs, gt_d)

    with open(os.path.join(cfg.work_dir, "models.pkl"), "wb") as fh:
        pickle.dump({"kinds": kinds, "models": models, "stacker": stacker, "features": feats}, fh)
    with open(os.path.join(cfg.work_dir, "decode.json"), "w") as fh:
        json.dump({"model": best_name, **best_cfg}, fh, indent=2, default=float)
    pd.DataFrame({"s1_id": df["s1_id"], "cand_id": df["cand_id"], "y": y, "p_stack": stack_oof,
                  **{f"p_{k}": v for k, v in oof.items()}}).to_parquet(
        os.path.join(cfg.work_dir, "oof_train.parquet"), index=False)
    save_report(cfg, "03_train", rep)


def breakdown(recs, s1_ids, per, pred, gt_d):
    country = recs.set_index("entity_id")["country"]
    t = pd.DataFrame({"s1": s1_ids, "f": per})
    t["country"] = country.reindex(t["s1"]).values
    t["gt_n"] = [len(gt_d.get(s, ())) for s in s1_ids]
    t["pred_n"] = [len(pred.get(s, ())) for s in s1_ids]
    sing = t["gt_n"] == 0
    return {
        "macro_f05_oof": round(float(t["f"].mean()), 5),
        "macro_f05_by_country": t.groupby("country")["f"].mean().round(5).to_dict(),
        "singletons": {"n": int(sing.sum()), "correct_empty_pct": round(100 * float((t.loc[sing, "pred_n"] == 0).mean()), 3)},
        "non_singletons": {"n": int((~sing).sum()), "mean_f": round(float(t.loc[~sing, "f"].mean()), 5),
                           "predicted_empty_pct": round(100 * float((t.loc[~sing, "pred_n"] == 0).mean()), 3)},
        "f_by_true_match_count": t.assign(g=t["gt_n"].clip(upper=5)).groupby("g")["f"].mean().round(4).to_dict(),
        "score_lost_by_bucket": {
            "false_merge_on_singleton": round(float((sing & (t["pred_n"] > 0)).sum()) / len(t), 5),
            "missed_all_matches": round(float((~sing & (t["pred_n"] == 0)).sum()) / len(t), 5),
            "partial": round(float(((1 - t["f"]) * (~sing & (t["pred_n"] > 0))).sum()) / len(t), 5)},
    }


def error_samples(recs, pdf, pred, gt_d, n=20):
    r = recs.set_index("entity_id")
    txt = lambda i: f"{r.at[i, 'business_name']} | {r.at[i, 'business_address']}" if i in r.index else i
    pred_pairs = {(s, c) for s, cs in pred.items() for c in cs}
    pdf = pdf.assign(pred=[(s, c) in pred_pairs for s, c in zip(pdf["s1_id"], pdf["cand_id"])],
                     true=[c in gt_d.get(s, ()) for s, c in zip(pdf["s1_id"], pdf["cand_id"])])
    fp = pdf[pdf["pred"] & ~pdf["true"]].sort_values("p", ascending=False).head(n)
    fn = pdf[~pdf["pred"] & pdf["true"]].sort_values("p", ascending=False).head(n)
    return {
        "n_false_positive_pairs": int((pdf["pred"] & ~pdf["true"]).sum()),
        "n_false_negative_pairs_in_candidates": int((~pdf["pred"] & pdf["true"]).sum()),
        "top_false_positives": [{"p": round(float(p), 3), "s1": txt(s), "cand": txt(c)}
                                for s, c, p in zip(fp["s1_id"], fp["cand_id"], fp["p"])],
        "top_false_negatives": [{"p": round(float(p), 3), "s1": txt(s), "cand": txt(c)}
                                for s, c, p in zip(fn["s1_id"], fn["cand_id"], fn["p"])],
    }


def leave_one_country_out(cfg, df, X, y, recs, gt_d):
    """Train LightGBM on all-but-one country, score the held-out one: proxy for France."""
    country = recs.set_index("entity_id")["country_key"]
    cvec = country.reindex(df["s1_id"]).values
    out = {}
    for c in sorted(set(cvec)):
        tr, va = cvec != c, cvec == c
        if tr.sum() == 0 or va.sum() == 0:
            continue
        with Timer(f"LOCO hold out {c!r}"):
            # small inner split of the training countries for early stopping
            idx = np.flatnonzero(tr)
            rng = np.random.RandomState(cfg.seed)
            s1u = np.unique(df["s1_id"].values[idx])
            es = set(rng.choice(s1u, size=max(1, len(s1u) // 10), replace=False))
            es_mask = np.array([s in es for s in df["s1_id"].values[idx]])
            m, _ = fit_lgb(cfg, X[idx[~es_mask]], y[idx[~es_mask]], X[idx[es_mask]], y[idx[es_mask]])
            p = m.predict(X[va], num_iteration=m.best_iteration)
        s1_ids = recs.loc[(recs["source"] == 1) & (recs["country_key"] == c), "entity_id"].tolist()
        pdf = pd.DataFrame({"s1_id": df["s1_id"].values[va], "cand_id": df["cand_id"].values[va], "p": p})
        b, _ = tune(pdf, gt_d, s1_ids, beta=cfg.beta, min_prob=cfg.min_prob)
        default = macro_f05(decode(pdf, "expected_f", exclusive=True, min_prob=cfg.min_prob), gt_d, s1_ids)
        out[c] = {"auc": round(roc_auc_score(y[va], p), 5), "f05_expected_f_default": round(default, 5),
                  "f05_best_tuned_on_itself": round(b["f05"], 5), "best": b}
    return out
