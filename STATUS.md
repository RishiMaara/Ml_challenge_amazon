# Amazon ML Challenge 2026: Business Entity Resolution STATUS

> Single source of truth for progress. Update the **Status** and **Notes** after every sub-step.
> Status legend: `TODO` · `IN PROGRESS` · `DONE` · `BLOCKED` · `AT RISK` · `SKIPPED`

## Dashboard
| Item | Value |
|---|---|
| Challenge window | ends **~27 Sep 2026 23:50–24:00 IST** (portal countdown read 2d 20h 03m at ~03:50 IST 25 Sep) |
| Internal freeze | **27 Sep 2026 21:00 IST** (final run, validate, zip, upload) |
| Results / Finale | 2 Oct 2026 / 7 Oct 2026 (virtual) |
| Best local F0.5 (val) | **0.9821 full val** (v5c = v4c refiner + blocking extension incl. Indian address codes) |
| Best public leaderboard F0.5 | **0.962** (v1 LightGBM); v3 0.957 |
| Blocking recall (val) / avg candidates per S1 | **95.79% / 54.5** (oracle F0.5 0.9855) |
| AWS spend so far | $0 of $160 |
| Current focus | Submit **v5c** at 26 Sep 00:00, read France from the LB, then France-only variants + final package. Submissions used 25 Sep: **5/5** (v3 0.957, one failed upload) |

## Approach at a glance
Blocking (union of exact keys + rare tokens + TF-IDF char kNN + multilingual embedding kNN + reverse kNN) → **LightGBM pairwise matcher** on similarity features → optional multilingual cross-encoder re-ranker (MIT/Apache, ≤ 8B params) → decision layer (calibration, one-S1-per-record assignment, expected-F0.5 subset selection, singleton handling).
Compute: laptop for EDA on samples; **AWS S3 + EC2** (r7i.8xlarge CPU, g5.2xlarge Spot GPU) for full scale; Kaggle/Colab GPU as the fallback.

---

## Step 0: Setup & infrastructure
| # | Sub-step | Status | Notes |
|---|---|---|---|
| 0.1 | Project layout (`code/business_entity_resolution/src`, `notebooks/`, `artifacts/`, `output/`) + `git init` | DONE | Layout created in submission structure; `.gitignore` excludes data/artifacts/.venv; git initialised (commit 0035fa0) |
| 0.2 | Python 3.11 venv + pinned requirements (polars, pyarrow, pandas, rapidfuzz, scikit-learn, lightgbm, faiss-cpu, sentence-transformers, torch, anyascii, jellyfish, boto3) | DONE | `.venv` (uv, CPython 3.11.15) at project root; pinned versions in `code/business_entity_resolution/requirements.txt` (torch 2.14 CPU build) |
| 0.3 | Install AWS CLI v2 | DONE | AWS CLI 2.37.1 at `%LOCALAPPDATA%\Programs\Amazon\AWSCLIV2` (already on user PATH; open a new terminal). Verified as `user/tricky` |
| 0.4 | File quota increases: EC2 On-Demand Standard → 64 vCPU; On-Demand G/VT → 8; G/VT Spot → 8 | SKIPPED | Both quota requests denied (account on AWS Free plan). **User decision 25 Sep: stay on the Free plan**, so no EC2 compute |
| 0.5 | AWS Budgets alerts ($40/$80/$120/$150) + tag `project=amlc2026` | DONE | Budget `amlc2026-credits` ($160, excludes credits so real burn is tracked); email alerts at $40/80/120/150 → gsreddy1182006@gmail.com |
| 0.6 | Create S3 bucket `amlc2026-856608368454` (ap-south-1); upload raw TSVs to `raw/` | DONE | Bucket `amlc2026-856608368454` (ap-south-1, private, tagged). `raw/` = 10 files / 2.4 GB (verified). `processed/` + `normalized/` backup upload running |
| 0.7 | Convert TSVs to parquet → `processed/` | DONE | `src/data.py` → `artifacts/processed/*.parquet` (zstd, 1.0 GB, 13 s) + `train_gt_pairs.parquet` (7.76M rows, exploded). S3 copy to `processed/` after the raw upload |
| 0.8 | Launch EC2 CPU box (r7i.8xlarge) + idle auto-stop alarm | SKIPPED | Free plan only allows Free-Tier instances. `infra/ec2_worker.py` kept for reference; the IAM role `amlc2026-ec2-worker` and SG `amlc2026-worker-sg` exist (no cost). **Compute = laptop (16 GB, 12 threads)** |
| 0.9 | GPU: g5.2xlarge Spot **or** Kaggle/Colab fallback ready | DONE | AWS GPU denied → **Kaggle** is the GPU path. Kaggle CLI 2.2.4 in .venv; user stored the token at `~/.kaggle/access_token` (auth verified 25 Sep 07:05) |

## Step 1: Understand problem & rules
| # | Sub-step | Status | Notes |
|---|---|---|---|
| 1.1 | Read problem statement | DONE | For each S1, list all matching S2/S3 IDs (0..n). S1 is deduplicated. |
| 1.2 | Understand the metric | DONE | Macro F0.5 per S1. Singleton: empty prediction → 1.0, anything else → 0.0. Implementation in 4.3 |
| 1.3 | Submission + validator rules | DONE | Two TSVs; one row per test S1; S2/S3 IDs only; no dupes; matches ⊆ candidates; `utils/validate_submission.py` |
| 1.4 | Compliance checklist | DONE | No external lookups/geocoding/internet data; model MIT/Apache ≤ 8B; reproducible zip; `country` is an open set |

## Step 2: Explore dataset (EDA)
| # | Sub-step | Status | Notes |
|---|---|---|---|
| 2.1 | Schema, sizes, nulls, duplicates | DONE | No duplicate entity_ids. S2/S3 have ~0.4–0.5% exact duplicate (name, addr, country) rows; S1 has none. Names never empty; S2/S3 addresses 2.7–3.4% empty |
| 2.2 | Country & source distributions | DONE | Train: US ~60%, India ~40%. Test adds **France** (S1 15%, S2/S3 ~14%) |
| 2.3 | Match-count distribution & singletons | DONE | Singletons 5.6%. Mode 3 matches; 0–11 range. Mean 1.67 S2 + 1.79 S3 per S1. ~2.7M S2/S3 distractors |
| 2.4 | Does each S2/S3 record map to ≤ 1 S1? | DONE | **YES**: all 7,638,365 matched S2/S3 ids map to exactly 1 S1. 26% of S2/S3 (2.68M) are unmatched distractors → assignment constraint valid |
| 2.5 | Do matches always share the `country` value? | DONE | **YES**: 100% of pairs share country (US–US 4.58M, India–India 3.06M) → partition blocking by country |
| 2.6 | Name noise catalogue | DONE | Permutations, OCR typos, accents, suffix swaps, appended descriptors, domain names (~4%), junk prefixes (~2–3%), alias constructions (D.B.A., t/a, formerly known as), native-script names, zero-width chars. See `docs/eda_findings.md` |
| 2.7 | Address noise catalogue | DONE | Reordering, dropped/altered house numbers, city swaps, NULL/PMB/PO Box injection, state abbreviation/full/native script. **Postcodes rare** (ZIP ~10% US, PIN ~0% India) → not a primary key |
| 2.8 | S2 vs S3 style differences | DONE | S2: US addresses 90% UPPERCASE with USPS abbreviations; more native script (India 41%). S3: full state names, title case, native script 33% (India) |
| 2.9 | Encoding & script inventory | DONE | Clean UTF-8. Indic scripts: Devanagari ≫ Telugu, Kannada, Tamil, Gujarati, Bengali, Malayalam, Oriya, Gurmukhi. Zero-width chars ~1%. French: accents, °/º, ’ |
| 2.10 | France test profile (unlabelled) | DONE | 3 regions / ~10 cities → huge city blocks; generic names (Club, Amicale, SARL…). S1 uses region, S2/S3 use department → need a department↔region map. Abbreviations R./AV/ALL/BD |
| 2.11 | Similarity of true matches vs random pairs | DONE | Sample of 60k: true name token-set median 95–100 vs random 33–38; address 93–97 vs 36–38. **Hard negatives are the issue**: 49% of S1 share a name key with another S1; 22% of distractors have an exact S1 name key |
| 2.12 | EDA findings summary (for Documentation §2.1) | DONE | Written to `docs/eda_findings.md` (feeds Documentation §2.1) |

## Step 3: Cleaning & normalisation
| # | Sub-step | Status | Notes |
|---|---|---|---|
| 3.1 | Unicode NFKC, accent folding, transliteration (anyascii) | DONE | `src/normalize.py`: NFKC, zero-width strip, anyascii transliteration, lowercase. **Learned translit map** from 1.1M train pairs: 539 tokens (praivet→pvt, limitet→ltd, indstrijh→industries…) + 16 native-script states. Indic names now normalise to exactly the S1 name (p10 token-set = 100) |
| 3.2 | Junk / URL / domain stripping | DONE | Pipe/URL tails stripped; domains and hashtags → concatenated core (`name_is_domain`); junk prefixes removed; OCR digit→letter fix; alias split (D.B.A., t/a, formerly known as, aka) → `name_core` + `name_alt` |
| 3.3 | Legal-suffix dictionaries (US, India, France) | DONE | `src/dictionaries.py` LEGAL_CANON (US/IN/FR) → `name_legal`; `name_core` excludes legal tokens + stopwords; `name_key` (sorted tokens), `name_skel` (consonant skeleton) |
| 3.4 | Address abbreviation expansion (US / IN / FR) | DONE | Country-specific street maps (US/India/France: R→rue, AV→avenue, ALL→allee…); number-prefix tokens dropped (H.No, Door No, Unit…); NULL/PMB/PO Box removed |
| 3.5 | Parse postcode / PIN, house number, city, state | DONE | `addr_street`, `addr_localities`, `addr_nums`, `addr_house_no`, `addr_postcode`, `addr_empty`. Train true pairs: house number agrees 88% US / 76% India (Indian addresses have many numbers → use set overlap) |
| 3.6 | State name ↔ abbreviation maps (incl. native scripts) | DONE | US 50 states + DC; India states/UTs + aliases (+Keralam); France 13 regions + all departments → region. State found 100% US / 99.3% India; agrees 100% / 98.7% on true pairs. France: S1 100%; S2 66% (the rest genuinely have no state) |
| 3.7 | Unit tests on tricky examples | DONE | `tests/test_normalize.py`: 17 tests from real EDA examples; all pass. Full run: 22M records in 13.5 min (6 workers). Output `artifacts/normalized/*.parquet` |

## Step 4: Validation framework
| # | Sub-step | Status | Notes |
|---|---|---|---|
| 4.1 | Group split by S1 (90/10), full S2/S3 pool kept as distractors | DONE | `src/split.py` → `artifacts/splits/split_v1.parquet`: 5 stratified folds (country × match-count), fold 0 = **val** (441,370 S1), 1–4 = **fit** (1,765,451); **dev** = 20k val subset. Singleton rate 5.59% in every part. **Blocking/features run over ALL train S1** (label-free); the split only selects whose labels fit vs score |
| 4.2 | Leave-one-country-out split (France proxy) | DONE | `split.subsets()`: `loco_India` (fit on US labels → score India val), `loco_US` (vice versa) |
| 4.3 | `evaluate.py`: macro F0.5, P, R, singleton accuracy | DONE | `src/evaluate.py`: exact macro F0.5 (singleton rule), macro/micro P/R, singleton accuracy, by-country breakdown; TSV read/write in submission format. Tests reproduce the README example = **0.714**; 23 tests pass |
| 4.4 | Blocking metrics: pair recall, entity recall ceiling, reduction ratio | DONE | `blocking_report()`: pair recall, entity full recall, **oracle F0.5 ceiling**, candidates per S1, reduction ratio, by country. Finding: exact name_key blocking → 28.6M pairs for val (Meridian = 1,725 records) with only 65.6% recall / 0.851 ceiling → name-only blocking unusable |
| 4.5 | Experiment log (`experiments.md`) | DONE | `artifacts/experiments.jsonl` → rendered to `docs/experiments.md`. References (dev): all-empty 0.055; exact name 0.467; +state 0.604; **+house no. 0.687 (P 0.98, R 0.48)** |

## Step 5: Blocking / candidate generation
| # | Sub-step | Status | Notes |
|---|---|---|---|
| 5.1 | Country partitioning (if 2.5 confirms) | DONE | Blocking runs per `country` (EDA 2.5: 100% of pairs share country). Open set: any new country gets its own partition |
| 5.2 | Exact-key blocks (postcode + name token, house no. + street token) | DONE | Composite keys as tokens: `wl` name word|locality, `hs` house no.|street word, `hk` house no.|street skeleton, `nk` full name key |
| 5.3 | Rare-token inverted index on names | DONE | IDF-weighted token overlap via joins (`src/blocking.py`); tokens with df > 3000 dropped. Name tokens: word, skeleton, word pairs, concatenation, 5-grams for domain names |
| 5.4 | TF-IDF char n-gram kNN | SKIPPED | Replaced by the token-overlap scorer (sparse TF-IDF equivalent; 5-grams cover char-level concatenation). Revisit only if recall stalls |
| 5.5 | Multilingual embedding kNN (FAISS) | DONE | Kaggle `amlc2026-embed-knn` (private; 2×T4, multilingual-e5-small MIT): top-6 neighbours per S1 per source. Val pair recall: blocking 95.79% → **union 96.67%** (US 97.78%, India 95.01%), +6.4 cands/S1. Union dirs `blocking/union_{train,test}` (hard links + `<country>_knn.parquet`) |
| 5.6 | Reverse kNN (S2/S3 → S1) | SKIPPED | Covered by the across-S1 competition features (rev_*) computed over all candidates |
| 5.7 | Union + dedupe; tune K for ≥ 98% pair recall | DONE | Full train (2.2M S1) in 68 min on the laptop → `artifacts/blocking/cands_all_df3000_t20n10a10/` (~121M pairs, 1.9 GB, integer ids). **Val (441k S1): pair recall 95.79% (US 96.8%, India 94.2%), entity full recall 87.4%, oracle F0.5 0.9855, 54.5 cands/S1**. The ordinal canonicalisation fix gave +0.1pp |
| 5.8 | Optional cheap pre-ranker to cap candidates per S1 | TODO | |
| 5.9 | Write `candidate_pairs.tsv` (final model input) | DONE | Test blocking 42 min: **93.2M pairs**. France 259k S1 → 52.4/S1, India 810k → 53.5, US 663k → 54.8. `artifacts/blocking/cands_test_df3000_t20n10a10/`. candidate_pairs.tsv is written from it at submission time |

## Step 6: Baseline submission
| # | Sub-step | Status | Notes |
|---|---|---|---|
| 6.1 | Rule / threshold baseline scored on validation | DONE | `src/baseline_rule.py`: name_key + house no. + compatible state, drop records claimed by >1 S1. **Val 0.6798** (P 0.989, R 0.467; US 0.727, India 0.610); dev 0.684 |
| 6.2 | Test run + validator PASS | DONE | Test: 2.93M pairs, 1.29M of 1.73M S1 non-empty. **Official validator PASS (with --check-ids)**. Empty rate France 19.6% / India 32.1% / US 20.0%; France has more matches per S1 (2.63 vs 2.15–2.25), a possible false-merge risk. Backed up to `s3://…/submissions/v0_baseline_rule/` |
| 6.3 | Upload to portal; record leaderboard score | DONE | Submitted 25 Sep 04:15 IST as **Team Bloom**: **public F0.5 0.680** (val 0.6798), so validation is well calibrated incl. France. Upload fix: Windows had no MIME type for .tsv, so the user set HKCU `.tsv` Content Type = text/tab-separated-values |

## Step 7: Feature engineering
| # | Sub-step | Status | Notes |
|---|---|---|---|
| 7.1 | Name metrics: ratio, token set/sort, Jaro-Winkler, Jaccard, TF-IDF cosine, IDF-weighted overlap | DONE | Name fuzzy set (ratio / partial / token-sort / token-set / JW / concat), equality, Jaccard, length diffs. ~65k pairs/s on the laptop |
| 7.2 | Phonetic & acronym match | DONE | Skeleton token-set, alias-vs-core, legal-form eq/Jaccard/missing, domain/indic flags |
| 7.3 | Address component match (postcode, house no., street, city, state) | DONE | Address/street/locality similarity, state/house/postcode agreement (null = missing), number-set Jaccard/shared, empty flag |
| 7.4 | Embedding cosines (name, address, combined) | DONE | `knn_cos`, `knn_rank` on union candidates; stage-1 val 0.9704 → 0.9720 with them |
| 7.5 | Missingness flags; source flag (S2/S3) | DONE | Missing → null; `src` flag; country excluded |
| 7.6 | Context: rank within S1, gap to best, reverse rank, #candidates | DONE | Within-S1 relative scores + across-S1 competition (rev_n_s1, rev_is_best, rev_margin, rev_score_rel, rev_name_rel) from ALL candidates of the split |
| 7.7 | Group consistency: similarity to other strong candidates | DONE | **Stage-2 group consistency** (`src/stage2.py`): compare each candidate with the S1's 2 best other candidates (p1, name/skeleton/address/house agreement). Full val **0.9732** vs stage-1 0.9704 (+0.0028), P 0.9952, R 0.9347 |
| 7.8 | Feature importance + leakage check | DONE | Gain v1: **rev_margin 0.66**, rev_is_best 0.10, a_nums_jacc 0.065, rev_score_rel 0.02, a_street_tset 0.02, a_tok_jacc 0.02. No label leakage: rev/ctx features use only blocking scores (label-free) |

## Step 8: Matching model
| # | Sub-step | Status | Notes |
|---|---|---|---|
| 8.1 | Build training pairs from train-split candidates (hard negatives) | DONE | Train = candidates of 200k random fit S1 (10.9M pairs, all negatives kept); valid = dev 20k (1.09M pairs) |
| 8.2 | LightGBM v1 | DONE | **LightGBM v1** (127 leaves, lr 0.08, 746 rounds, 4.7 min) + one-owner + thr 0.70: **dev F0.5 0.9706** (P 0.9937, R 0.9303; US 0.9752, India 0.9637; singleton acc 0.969). `artifacts/models/lgb_v1.txt` |
| 8.3 | Hyperparameter tuning | DONE | v2 full val **0.9710** @thr 0.70/0.75 (US 0.9762, IN 0.9631; crowded slice 0.9702). Expected-F0.5 / hybrid / name floor all ≤0.9705, so plain threshold kept |
| 8.4 | Probability calibration | SKIPPED | LightGBM logloss probabilities are well-behaved; the threshold curve is flat 0.65–0.80 |
| 8.5 | Leave-one-country-out robustness check | DONE | LOCO: US→India blind **0.9016** vs in-domain 0.9624 (−0.061); India→US blind 0.9623 vs 0.9756 (−0.013). Threshold re-tuning adds ≤0.01. The implied France score ≈0.93 exceeds the generic cost, pointing to the France-specific "several businesses at one address" false merges |

## Step 9: Decision layer / post-processing
| # | Sub-step | Status | Notes |
|---|---|---|---|
| 9.1 | Global threshold sweep for F0.5 | TODO | |
| 9.2 | One-S1-per-record assignment | TODO | Depends on 2.4 |
| 9.3 | Expected-F0.5 subset selection per entity | DONE | Expected-F0.5 prefix selection tested: best 0.9705 (temp 1.5) < threshold 0.9710; the model is already sharp. Kept for documentation |
| 9.4 | Singleton handling ("predict empty" option) | DONE | Singletons handled by threshold + one-owner (singleton acc 0.970 @0.75); expected-F empty option was worse (0.91–0.95) |
| 9.5 | Ablation of each component | DONE | Ablation (val): threshold 0.9710 > expected-F 0.9705 > hybrid 0.9699 > +name floor 40/50/60 0.9689/0.9679/0.9666 |

## Step 10: Neural components & ensemble
| # | Sub-step | Status | Notes |
|---|---|---|---|
| 10.1 | Compute embeddings on GPU (EC2 or Kaggle) | DONE | Embeddings on Kaggle GPU (24M texts, ~14 min per split on 2×T4); chunked fp16 + tiled exact top-k (fixes: __main__ guard, RAM OOM, GPU OOM) |
| 10.2 | Cross-encoder fine-tune (stretch goal) | TODO | Only if ahead of schedule on Day 3 AM |
| 10.3 | Blend / stack with LightGBM | TODO | |
| 10.4 | Licence + parameter-count check | TODO | |

## Step 11: France generalisation
| # | Sub-step | Status | Notes |
|---|---|---|---|
| 11.1 | French legal-form + address dictionaries | DONE | French legal forms, street types, department→region map in `dictionaries.py` |
| 11.2 | Country-agnostic feature audit | DONE | Crowd features were out-of-distribution for France (3× crowding) → **removed from v3** |
| 11.3 | Manual review of a sample of French predictions | IN PROGRESS | v2 LB drop shows France has **sibling businesses** (same core name + descriptor, same street, different house no.) as hard negatives; v1 France matches have house-no. conflicts in only 1.5% of pairs (US 9.8%, IN 19.5%). Probe p1 tests removing them |
| 11.4 | Optional pseudo-labelling of high-confidence French pairs | TODO | |

## Step 12: Full-scale test inference & submissions
| # | Sub-step | Status | Notes |
|---|---|---|---|
| 12.1 | End-to-end run on EC2 | DONE | v3 test: union features 104M pairs → stage-2 → safeguard (rejected 185k French house-conflict pairs) → 5,679,374 pairs, 6.3% empty. France 7.0% empty / 3.42 per S1 (v1: 4.8% / 3.67); US/IN slightly more matches |
| 12.2 | Validator PASS | DONE | v3 matching_results.tsv **validator PASS (--check-ids)**; `submissions/v3u/` (+ `v3u_code.zip`); fallback no-kNN v3 in `submissions/v3/` |
| 12.3 | Submission log (see table below) | IN PROGRESS | v0 0.680, **v1 0.962** (dev 0.9706). Gap ≈ dev-threshold optimism + France: if US/IN ≈ 0.970 then France ≈ 0.91 → France is the biggest lever |

## Step 13: Error analysis & iteration
| # | Sub-step | Status | Notes |
|---|---|---|---|
| 13.1 | False-positive review | DONE | Val FPs 7.4k pairs, 95% are distractors. Distractor recipe (all countries): S1 name + extra word / swapped word / legal form change, **house number shifted +1..+21 on the same street**. Distractors are one-offs (1.6% share name+house with another record vs 55% of true copies) and almost never have an empty address (0.3%). Test has 5.76 records/S1 vs 4.67 train → ~2x distractors per S1 on test (precision drops vs val) |
| 13.2 | False-negative review | DONE | v3 val 0.9748: fix low-p in-candidate misses → 0.9834; remove all FPs → 0.9791; perfect within candidates → 0.9889. Half the low-p misses are empty-address records (97.7% of empty-address records are true matches; ambiguous when several S1 share the name). Blocking misses 50.9k (3.3%): many easy (same address + name typo) cut by the per-S1 top-k caps. No row-order / ID leaks. French house-number parse bug (postcodes '59200 Tourcoing', 'Appartement 406', '2eme etage') |
| 13.3 | Fixes + prioritised re-runs | IN PROGRESS | v4 `src/refine.py` + `notebooks/run_v4.py`: stage-3 LightGBM on v3 features + p_v3 + house_v2 (postcode/apartment aware) delta/lev/substring + name-token miss/extra + copy-support counts; 2-fold on val. Next: address-anchored blocking extension (v5) |

## Step 14: Final package
| # | Sub-step | Status | Notes |
|---|---|---|---|
| 14.1 | `src/` cleanup + single `run_all` entry point | TODO | |
| 14.2 | `README.md` (reproduce end-to-end) + pinned `requirements.txt` | TODO | |
| 14.3 | Fill in `Documentation_template.md` | TODO | Write incrementally from Day 1 |
| 14.4 | Zip structure check (`<team>_submission.zip`) | TODO | |
| 14.5 | Final upload before 18:00 IST, 27 Sep | TODO | |

## Step 15: AWS cost control & teardown
| # | Sub-step | Status | Notes |
|---|---|---|---|
| 15.1 | Daily spend check | TODO | |
| 15.2 | Stop idle instances | TODO | |
| 15.3 | Final teardown; archive artefacts in S3 | TODO | |

---

## Submission log
| # | Date/time (IST) | Change | Local val F0.5 | Public LB F0.5 |
|---|---|---|---|---|
| v0 | 25 Sep 04:15 | Exact rule: name_key + house no. + state; candidates = matches | 0.6798 | **0.680** |
| v1 | 25 Sep ~06:15 | LightGBM v1 (200k fit S1) + one-owner + thr 0.70 | 0.9706 (dev) | **0.962** |
| v2 | 25 Sep ~08:30 | + crowding / name-conflict / within-S1 name-rank features, thr 0.75 | 0.9710 (full val) | **0.951** ❌ (France ≈0.85: crowd features out-of-distribution; added 35k same-street, different-house-no. "sibling" merges) |
| p1 | — | PROBE (not submitted: user budget = 5/day, no probes) | = v1 | — |
| v3 | 25 Sep 15:06 | stage-1 (no crowd, +knn_cos) + stage-2 group consistency + 600k fresh fit S1 + Kaggle neighbours (union candidates) + unseen-country house-number safeguard; thr 0.70 | **0.9748** (full val; no-kNN variant 0.9732) | **0.957** (0.956647) |
| v4 | file ready (`submissions/v4`) | stage-3 refiner on v3 (house_v2 delta/lev/substring, name-token miss/extra, copy-support counts), 2-fold on val; no blanket France safeguard; thr 0.75 | 0.9773 (OOF full val) | — (fallback) |
| v4b | discarded | + acronym + per-split token frequency | 0.9785 | ❌ not submitted: token-frequency values differed train vs test (27.32 vs 27.45) → tree thresholds flipped, dropped 46k India test pairs |
| v4c | file ready (`submissions/v4c` + `v4c_code.zip`), **planned upload 26 Sep 00:00** | v4b with train-referenced, log2-bucketed token frequency | **0.9781** (OOF full val; US 0.9829, IN 0.9709) | pending |
| v5 | file ready (`submissions/v5`) | v4c + blocking extension (house+street / name+street / house+name keys, top-3 per S1) scored by its own 2-fold LightGBM; ext thr 0.6 | 0.9798 (IN 0.9738) | — |
| v6 | discarded | + name-only extension for unclaimed empty-address records | 0.9798 (no gain) | — |
| v5b | file ready (`submissions/v5b`) | extension + Indian address-code keys (B-46, D-2/201 with state / locality) + exact name key + state, top-5 per S1 | 0.9819 (US 0.9840, IN 0.9787) | — (fallback) |
| v5c | **file ready (`submissions/v5c` + `v5c_code.zip`) — planned upload 26 Sep 00:00** | v5b + rare adjacent address-word-pair key; main thr 0.75, ext thr 0.7; hashed per-country key builder (fits 16 GB) | **0.9821** (US 0.9842, IN 0.9789) | pending |

## Decision log
| Date | Decision | Why |
|---|---|---|
| 25 Sep | Hybrid: blocking + LightGBM + optional cross-encoder + F0.5 decision layer | Tabular pair matching favours GBDT; embeddings handle scripts/reorders; the decision layer optimises the exact metric |
| 25 Sep | EC2 (CPU + GPU Spot) + S3; Kaggle/Colab GPU fallback | SageMaker quotas are 0 and it costs more; EC2 is the simplest way to use the $160 |
| 25 Sep | Block by country; no postcode-based primary key; one-S1-per-record assignment | EDA 2.4/2.5/2.7 confirmed |
| 25 Sep | Learn native-script→Latin maps from train pairs (positional alignment) instead of hand-writing them | Covers 9 Indic scripts; derived only from training data (no external data) |
| 25 Sep | Validation: blocking/features over all train S1; split only selects labels for fit (80%) vs val (20%) + 20k dev; LOCO as France proxy | Keeps S1-vs-S1 competition for each record realistic, as on test |
| 25 Sep | GPU work moves to Kaggle; AWS GPU appeal filed at a reduced size | AWS denied the G/VT quota for this new account |
| 25 Sep | Stay on the AWS Free plan: S3 only; compute on laptop + Kaggle | User decision; avoids card billing beyond credits |
| 25 Sep | Every portal submission = matching_results.tsv + code zip (`submissions/vN_code.zip`) | The portal form needs both files to enable Submit & Evaluate |
| 25 Sep | Drop crowd features (France out-of-distribution); use LB probes that change ONLY France rows to test France hypotheses | v2 val +0.0007 but LB −0.011 |
| 25 Sep | No git commits by the assistant (user commits); max 5 submissions/day, no probes | User instruction |
| 25 Sep | Unseen-country safeguard: reject house-number conflicts for countries absent from training | v2 regression + v1 France pattern (1.5% conflicts) |
| 25 Sep | Treat France as a first-class target (country-agnostic features, French dictionaries, LOCO validation) | 15% of test S1, zero training labels |
