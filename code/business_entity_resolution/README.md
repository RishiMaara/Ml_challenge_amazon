# Business Entity Resolution: Amazon ML Challenge 2026

Pipeline: raw TSV → parquet → normalisation → blocking (candidate generation) → matching → `output/*.tsv`.
Only the provided training/test data is used. No external lookups, APIs or geocoding (see *Compliance*).

## Environment
```bash
python -m venv .venv && .venv/Scripts/activate        # Python 3.11
pip install -r requirements.txt
```
The pipeline runs on a 16 GB / 12-thread laptop. All heavy steps are chunked.

## Reproduce (run from this folder; paths assume the repo layout)
```bash
# 1. TSV -> parquet (13 s)
python -m src.data --raw-dir ../../student_resource/dataset --out-dir ../../artifacts/processed
# 2. Normalisation. Learns native-script transliteration maps from TRAIN pairs only, then normalises
#    all 6 files (~14 min)
python -m src.normalize --processed ../../artifacts/processed --out ../../artifacts/normalized --workers 6
# 3. Validation split (train S1 entities: fit / val / dev / leave-one-country-out)
python -m src.split --processed ../../artifacts/processed --out ../../artifacts/splits
# 4. Blocking: candidates per S1 (train: ~68 min for 2.2M S1; test similar)
python -m src.blocking --norm ../../artifacts/normalized --split-file ../../artifacts/splits/split_v1.parquet \
    --gt ../../artifacts/processed/train_gt_pairs.parquet --subset all --out ../../artifacts/blocking
python -m src.blocking --norm ../../artifacts/normalized --split-file ../../artifacts/splits/split_v1.parquet \
    --subset test --out ../../artifacts/blocking
# 5. (GPU, optional; run on Kaggle) multilingual-e5-small embeddings + exact top-6 cosine neighbours per S1/source
python -m src.embed_export --norm ../../artifacts/normalized --out ../../artifacts/kaggle_export   # upload as dataset
#    run kaggle/embed_knn/embed_knn.py on 2xT4 -> knn_train.parquet, knn_test.parquet (~1 h)
# 6. Union candidates = blocking + embedding neighbours
python -m src.knn --knn ../../artifacts/kaggle_out/knn_train.parquet --split train \
    --cands ../../artifacts/blocking/cands_all_df3000_t20n10a10 --out ../../artifacts/blocking/union_train
python -m src.knn --knn ../../artifacts/kaggle_out/knn_test.parquet --split test \
    --cands ../../artifacts/blocking/cands_test_df3000_t20n10a10 --out ../../artifacts/blocking/union_test
# 7. Pair features (subsets: fit_sample = stage-1 train, fit_sample2 = 600k fresh S1 for stage 2, dev, val, test)
for s in fit_sample dev val; do python -m src.features --split train --cands ../../artifacts/blocking/union_train \
    --subset $s --knn ../../artifacts/kaggle_out/knn_train.parquet --out ../../artifacts/features/${s}_u; done
python -m src.features --split train --cands ../../artifacts/blocking/union_train --subset fit_sample2 --fit-sample 600000 \
    --knn ../../artifacts/kaggle_out/knn_train.parquet --out ../../artifacts/features/fit_sample2_u
python -m src.features --split test --cands ../../artifacts/blocking/union_test \
    --knn ../../artifacts/kaggle_out/knn_test.parquet --out ../../artifacts/features/test_u
# 8. Stage 1 + stage-2 group consistency + val report (see notebooks/run_v3.py)
python ../../notebooks/run_v3.py --tag u --retrain-s1
# 9. Test: group features -> stage-2 scores -> decision layer (unseen-country safeguard, one owner, thr 0.70)
python -m src.stage2 --stage1 ../../artifacts/models/lgb_s1_u.txt --features ../../artifacts/features/test_u \
    --split test --out ../../artifacts/features/test_s2u
python -m src.predict --features ../../artifacts/features/test_s2u --model ../../artifacts/models/lgb_v3_u.txt \
    --out ../../output --cands ../../artifacts/blocking/union_test --write-candidates
# 10. Validate
cd ../../student_resource && python utils/validate_submission.py --matching ../output/matching_results.tsv \
    --candidate ../output/candidate_pairs.tsv --test-dir dataset/test
```
(`src.baseline_rule` reproduces the v0 exact-rule baseline.)

## Modules (`src/`)
| Module | Purpose |
|---|---|
| `data.py` | TSV → parquet; ground truth exploded to (s1_id, match_id) pairs |
| `dictionaries.py` | Hand-written language dictionaries: legal forms (US/IN/FR), street types, states, French department→region |
| `normalize.py` | Unicode/zero-width cleanup, transliteration, alias split (D.B.A./t/a), legal-form extraction, address parsing (state, house no., street, localities); learns the native-script→Latin token map from train pairs |
| `split.py` | Stratified folds over train S1; dev subset; leave-one-country-out subsets |
| `evaluate.py` | Exact macro F0.5 (singleton rule), blocking report (pair recall, oracle F0.5), TSV IO, experiment log |
| `blocking.py` | IDF-weighted token-overlap blocking (name words, skeletons, word pairs, concatenations, 5-grams, name×locality, house×street), top-K union of total/name/address rankings per source |
| `baseline_rule.py` | v0 exact-rule matcher (name key + house no. + state) |
| `embed_export.py` | Compact text export (normalised name + city) for the Kaggle embedding job |
| `knn.py` | Merges embedding neighbours: recall report, union candidate dirs, `knn_cos`/`knn_rank` features |
| `features.py` | ~65 pair features: name/address similarities, agreement flags, blocking scores, within-S1 and across-S1 competition, name-conflict and within-S1 name rank, embedding similarity |
| `train.py` | LightGBM training utilities, one-owner decision, threshold tuning |
| `stage2.py` | Group consistency: compares each candidate with the S1's best other candidates (stage-1 probabilities) |
| `decision.py` | Decision strategies (threshold, expected-F0.5); threshold won on validation |
| `predict.py` | Test scoring + decision layer incl. the unseen-country house-number safeguard; writes both TSVs |

Model: two LightGBM binary classifiers (MIT licence, ~1k trees each). The embedding model `intfloat/multilingual-e5-small`
(MIT, 118M parameters) is used only to retrieve candidates and compute a similarity feature.

Tests: `python -m pytest tests -q`

## Compliance
- Data: only the challenge's train/test files. The transliteration map is learned from the training pairs.
- Dictionaries in `dictionaries.py` are general language knowledge (abbreviations, state/region names), not business data.
- No external APIs, databases or geocoding.
