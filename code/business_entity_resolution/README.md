# Business Entity Resolution — pipeline

Blocking → pairwise features → GBDT ensemble (LightGBM + XGBoost + CatBoost, LR stacker)
→ per-entity expected-F0.5 decoding → `matching_results.tsv` + `candidate_pairs.tsv`.

The design and the reasoning behind each stage: [`docs/SOLUTION_DESIGN.md`](../../docs/SOLUTION_DESIGN.md).

## Data handling

* The raw challenge TSVs are **read-only**. They're read with `sep="\t"` and `QUOTE_NONE`,
  because addresses contain commas and quotes. Nothing converts or rewrites them.
* Every derived table goes to `work/` as Parquet: normalized records, candidates, features,
  out-of-fold predictions, test scores and models. These files are large and are not committed.
* The two submission files are written to `output/` as TSV, exactly as the rules specify.
* Each stage writes a small JSON report to `reports/`. **Those reports are what you send back for tuning.**

## Setup (Windows, PowerShell)

```powershell
cd E:\AmazonMl
git clone -b claude/dreamy-gauss-5yhrg3 https://github.com/RishiMaara/Ml_challenge_amazon.git repo
cd repo\code\business_entity_resolution
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$D = "E:\AmazonMl\student_resource\dataset"     # folder containing train\ and test\
```

## Step 1 — smoke test (≈5–15 min)

Runs the whole pipeline on 3,000 real S1 records per split, matched against all of S2/S3.
It only checks that everything runs on your data. Output goes to `work_smoke/`, `reports_smoke/`
and `output_smoke/`. The validator will report missing S1 rows at the end, which is expected
for a sample.

```powershell
python -m src.run profile --data-dir $D
python -m src.run all --data-dir $D --set sample_s1=3000 --set use_catboost=false --no-loco
```

## Step 2 — full run, one stage at a time

```powershell
python -m src.run block    --data-dir $D     # -> reports\01_blocking_train.json, 01_blocking_test.json
python -m src.run features --data-dir $D     # -> reports\02_features_train.json, 02_features_test.json
python -m src.run train    --data-dir $D     # -> reports\03_train.json   (slowest stage)
python -m src.run predict  --data-dir $D     # -> output\*.tsv, reports\04_predict.json, runs the validator
```

Change any setting without editing code by adding `--set key=value` (all keys are in `src/config.py`), for example:
`--set cap_per_source=30 --set k_name_char=30 --set use_catboost=false --set n_jobs=6`.

## What to send back after each run

Paste the JSON files from `reports/` (they're small), plus any traceback. The most important ones:

| Report | What gets tuned from it |
|---|---|
| `00_profile.json` | Normalization rules, the exclusivity assumption, cross-country blocking, singleton handling |
| `01_blocking_train.json` | Retriever `k` values and `cap_per_source` (recall vs. pairs), new blocking keys, based on the missed-pair samples |
| `03_train.json` | Features (from the false-positive/false-negative samples and importances), model settings, decoding rule, and generalization to the unseen country (leave-one-country-out) |
| `04_predict.json` | Drift check on the test data: match rate and probability distribution per country, including France |

## If you run out of memory

* `--set chunk_rows=5000` — smaller blocking chunks
* `--set feature_chunk=50000 --set n_jobs=4` — smaller feature chunks and fewer worker processes
* `--set cap_per_source=15` — fewer candidate pairs (check recall in `01_blocking_train.json` first)
* `--set use_catboost=false` — CatBoost is the slowest model on CPU

## Code map

| File | Stage |
|---|---|
| `src/config.py` | All settings |
| `src/io_utils.py` | TSV reading/writing, reports |
| `src/normalize.py` | Unicode folding, legal-suffix/address abbreviation tables (EN/IN/FR), postcode/house-number/landmark extraction |
| `src/profile_data.py` | Stage 0: normalization and data profile |
| `src/blocking.py` | Stage 1: multi-retriever blocking, recall report |
| `src/features.py` | Stage 2: pairwise and relational features |
| `src/train.py` | Stage 3: cross-validated GBDT ensemble, stacker, decoding search, leave-one-country-out, error analysis |
| `src/decode.py` | Exact macro-F0.5 metric, expected-F0.5 set decoding, exclusivity |
| `src/predict.py` | Stage 4: scoring the test set, writing the submission, validation |
| `src/run.py` | Command-line entry point |
