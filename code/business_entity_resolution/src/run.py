"""Command-line entry point.

    python -m src.run profile  --data-dir E:\\AmazonMl\\student_resource\\dataset
    python -m src.run block    --data-dir ...          (train + test)
    python -m src.run features --data-dir ...          (train + test)
    python -m src.run train    --data-dir ...
    python -m src.run predict  --data-dir ...
    python -m src.run all      --data-dir ...

Override any config value with --set, e.g.  --set cap_per_source=30 --set use_catboost=false
"""

import argparse
import json
import os
import time

from .config import load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["profile", "block", "features", "train", "predict", "all"])
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--split", choices=["train", "test", "both"], default="both")
    ap.add_argument("--no-loco", action="store_true", help="skip leave-one-country-out evaluation")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = ap.parse_args()
    cfg = load_config(args.set, args.data_dir)
    cfg.save(f"{cfg.report_dir}/config_used.json")
    splits = ["train", "test"] if args.split == "both" else [args.split]
    t0 = time.time()
    times = {}

    def stage(name, fn):
        t = time.time()
        fn()
        times[name] = round((time.time() - t) / 60, 1)

    if args.stage in ("profile", "all"):
        from . import profile_data
        stage("profile", lambda: profile_data.run(cfg))
    if args.stage in ("block", "all"):
        from . import blocking
        for s in splits:
            stage(f"block_{s}", lambda s=s: blocking.run(cfg, s))
    if args.stage in ("features", "all"):
        from . import features
        for s in splits:
            stage(f"features_{s}", lambda s=s: features.build(cfg, s))
    if args.stage in ("train", "all"):
        from . import train
        stage("train", lambda: train.run(cfg, loco=not args.no_loco))
    if args.stage in ("predict", "all"):
        from . import predict
        stage("predict", lambda: predict.run(cfg))
    summary(cfg, times, time.time() - t0)


def summary(cfg, times, total):
    def load(name):
        p = os.path.join(cfg.report_dir, f"{name}.json")
        return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else {}
    b, t = load("01_blocking_train"), load("03_train")
    print("\n" + "=" * 60)
    print("SUMMARY")
    if t:
        print(f"  VALIDATION F0.5 (macro, out-of-fold) : {t.get('macro_f05_oof')}")
        print(f"  F0.5 by country                      : {t.get('macro_f05_by_country')}")
        print(f"  best model / decoding                : {t.get('chosen')}")
    if b:
        print(f"  blocking recall (capped)             : {b.get('recall_at_cap', {}).get(str(cfg.cap_per_source), {}).get('recall')}")
        print(f"  F0.5 ceiling if matcher were perfect : {b.get('macro_f05_ceiling_capped')}")
    for k, v in times.items():
        print(f"  time {k:<32}: {v} min")
    print(f"  TOTAL TIME                           : {total / 60:.1f} min")
    print("=" * 60)


if __name__ == "__main__":
    main()
