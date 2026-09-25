"""Central configuration. Every tunable knob lives here so a tuning round is a
one-line change. Override any value from the CLI with --set key=value."""

from dataclasses import dataclass, field, asdict
import json
import os


@dataclass
class Config:
    # ---------------- paths ----------------
    data_dir: str = "../../student_resource/dataset"   # folder holding train/ and test/
    records_dir: str = "work"                           # normalized records (shared by full + smoke runs)
    work_dir: str = "work"                              # Parquet intermediates (large, not committed)
    report_dir: str = "reports"                         # small JSON reports -> paste these back
    output_dir: str = "output"                          # matching_results.tsv / candidate_pairs.tsv

    # ---------------- runtime ----------------
    n_jobs: int = max(1, (os.cpu_count() or 2) - 1)
    seed: int = 42
    sample_s1: int = 0               # >0: smoke run on this many real S1 records per split (all S2/S3 kept)
    chunk_rows: int = 20000          # S1 rows per sparse top-k chunk
    feature_chunk: int = 200000      # pairs per feature worker chunk

    # ---------------- blocking ----------------
    # Retrieval runs per (country, target source) so S3 candidates never get crowded out by S2.
    k_name_char: int = 20            # top-k from name char 3-4gram TF-IDF
    k_name_word: int = 10            # top-k from name word 1-2gram TF-IDF
    k_full_word: int = 20            # top-k from name+address word TF-IDF
    k_addr_char: int = 10            # top-k from address char n-gram TF-IDF
    max_key_block: int = 50          # exact-key blocks larger than this are skipped (chains)
    min_score: float = 0.05          # cosine floor for sparse retrieval
    # after union: keep this many per (S1, source) ranked by a cheap combined score
    cap_per_source: int = 25

    # ---------------- model ----------------
    n_folds: int = 5
    use_catboost: bool = True
    use_xgboost: bool = True
    lgb_rounds: int = 3000
    lgb_lr: float = 0.03
    lgb_leaves: int = 127
    early_stop: int = 100
    neg_per_s1: int = 0              # 0 = keep all negatives; >0 = subsample easy negatives

    # ---------------- decoding ----------------
    decode: str = "expected_f"       # "expected_f" or "threshold"
    threshold: float = 0.5           # used when decode == "threshold" (re-tuned on OOF)
    exclusive: bool = True           # each S2/S3 id may belong to at most one S1 (checked in profile)
    min_prob: float = 0.02           # candidates below this are ignored by the decoder
    beta: float = 0.5

    extra: dict = field(default_factory=dict)

    def path(self, *parts):
        return os.path.join(*parts)

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)


def load_config(overrides=None, data_dir=None):
    cfg = Config()
    if data_dir:
        cfg.data_dir = data_dir
    for item in overrides or []:
        key, _, raw = item.partition("=")
        if not hasattr(cfg, key):
            raise SystemExit(f"Unknown config key: {key}")
        cur = getattr(cfg, key)
        if isinstance(cur, bool):
            val = raw.lower() in ("1", "true", "yes")
        elif isinstance(cur, int):
            val = int(raw)
        elif isinstance(cur, float):
            val = float(raw)
        else:
            val = raw
        setattr(cfg, key, val)
    if cfg.sample_s1 > 0:  # keep smoke-run artifacts away from the full run
        cfg.work_dir, cfg.report_dir, cfg.output_dir = "work_smoke", "reports_smoke", "output_smoke"
    for d in (cfg.records_dir, cfg.work_dir, cfg.report_dir, cfg.output_dir):
        os.makedirs(d, exist_ok=True)
    return cfg
