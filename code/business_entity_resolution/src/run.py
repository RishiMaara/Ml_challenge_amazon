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

    if args.stage in ("profile", "all"):
        from . import profile_data
        profile_data.run(cfg)
    if args.stage in ("block", "all"):
        from . import blocking
        for s in splits:
            blocking.run(cfg, s)
    if args.stage in ("features", "all"):
        from . import features
        for s in splits:
            features.build(cfg, s)
    if args.stage in ("train", "all"):
        from . import train
        train.run(cfg, loco=not args.no_loco)
    if args.stage in ("predict", "all"):
        from . import predict
        predict.run(cfg)
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
