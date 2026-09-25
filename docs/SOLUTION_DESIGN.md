# Business Entity Resolution — Solution Design

Status: **v1 implemented** in `code/business_entity_resolution/`. Not yet run on the real data.
Every impact figure below is an *expectation to verify*. The pipeline's reports measure each
one, and the plan changes when the numbers disagree.

---

## 0. What the metric rewards (this drives every decision)

* **F0.5 is computed per Source-1 entity and then macro-averaged.** A pair-level F0.5 is a
  different number. One wrong match on an entity that has one true match drops that entity from 1.0
  to 0.556 (precision 1/2, recall 1). A missed match drops it to 0.
* **Singletons count.** A Source-1 entity with no true match scores 1.0 for an empty
  prediction and 0 for any prediction. If 40% of S1 are singletons, 40% of the score depends
  on staying silent. `00_profile.json` reports the real singleton rate.
* **Consequence:** the decision is *which set to output per S1 entity*, not *which pairs pass a
  global threshold*. The pipeline outputs, for each entity, the set with the highest **expected
  F0.5** under the model's probabilities, where the empty set is one of the options
  (§10). A global threshold sweep is still run as a baseline, and whichever scores higher on
  out-of-fold data is used.
* **The test set contains a country that isn't in training (France).** Every stage is
  country-agnostic. Generalization is measured with **leave-one-country-out** validation (§11), not only
  a random split.

### Where this design differs from the proposed stack, and why

| Proposal | Change | Reason |
|---|---|---|
| Global threshold search 0.60–0.90 | Per-entity expected-F0.5 set decoding; the threshold stays as a baseline | The metric is per entity and includes singletons. The best cutoff depends on the other candidates in the same entity's list. |
| LightGBM *ranker* | Binary classifier plus relational rank/margin features | Set decoding needs calibrated probabilities. Ranking objectives don't produce them. The rank/margin features capture the ranking information anyway. |
| Explicit hard-negative mining stage | Train on the blocking output itself; add competition features | The top-25 retrieved non-matches per source *are* the hardest negatives, and training on them matches the inference distribution exactly (§8). |
| BGE-small embeddings | Multilingual model (multilingual-e5-small, MIT), and only if the missed-pair analysis shows semantic misses | BGE-small-en is English-only and the hidden test country is France. On typo/abbreviation noise, character n-grams usually already find the pair. |
| "Reduction ratio > 95%" | Target is *candidates per S1 entity* at ≥99% recall | At about 10⁵–10⁶ records per source, any blocking scheme exceeds a 99.99% reduction ratio. The real cost driver is pairs per entity. |

---

## 1. Data cleaning & normalization — `normalize.py` (implemented)

* Unicode NFKD with combining marks removed (é→e, ç→c). `&`→`and`. Punctuation becomes a space. Runs of single
  letters are merged (`A.B.C.`→`abc`, `P V T`→`pvt`).
* **Legal-suffix canonicalization** covering English, Indian and French forms: private/pvt/pte→`pvt`,
  limited→`ltd`, SARL/SAS/SASU/EURL/SA, GmbH and others. This produces two strings: `name_norm` (keeps legal
  words) and `name_core` (drops them). So `Star Coffee Pvt Ltd` and `Star Coffee Private Limited`
  both become the core `star coffee`, while `Star Coffee Roasters` keeps `roasters`.
* **Address canonicalization**: street types (road/rd/marg/salai→`rd`, avenue/av→`ave`,
  boulevard/bd→`blvd`, rue, chemin, allée), units (suite, floor, étage, bâtiment), landmark words
  (near/opp/behind/près), directions.
* **Structured extraction** from the raw address: postcode (last 5–6 digit group in the back
  part of the address, so a leading 5-digit US house number isn't mistaken for a ZIP; split Indian PINs `560 001` are
  merged; ZIP+4 is reduced to ZIP5), house/unit numbers, landmark phrase, and the last three words (usually city/state).
* Nothing keys on the country label. The country only defines blocking partitions.

| | |
|---|---|
| Why it helps | Every later similarity score measures real differences, not formatting differences |
| Precision | + (the unmatched-token features in §4 only work if legal words are removed first) |
| Recall | ++ (blocking keys and TF-IDF vocabularies line up across sources) |
| F0.5 | + |
| Cost | One pass, a few minutes for 1.7M records; cached as Parquet |
| Risks | Over-normalization merges distinct businesses (e.g. `Sun Pharma Ltd` vs `Sun Pharma Inc`). **Mitigation:** the legal-form mismatch feature (`n_legal_eq`) and `name_norm`-based features keep that signal available to the model. French coverage can't be tuned on labels, so `00_profile.json` prints normalized France samples for review. |

## 2. Blocking / candidate generation — `blocking.py` (implemented)

Within each country partition, and **separately for each target source** (S2, S3):

| Retriever | Catches |
|---|---|
| `name_char`: char 3–4-gram TF-IDF on `name_core`, top-20 | typos, spacing, transliteration (`Shree`/`Sri`) |
| `name_word`: word 1–2-gram TF-IDF, top-10 | word reordering, dropped words |
| `full_word`: word TF-IDF on name + address, top-20 (BM25-like weighting via IDF and sublinear TF) | trade/DBA names with the same address |
| `addr_char`: char n-gram TF-IDF on the address, top-10 | name rewritten, address kept |
| Exact keys: `name_core`; `postcode + first name token`; `acronym + postcode`, blocks capped at 50 | exact but rare-token matches that cosine similarity ranks low |

The hits are merged with **reciprocal-rank fusion**, and then the top `cap_per_source` (25) per (S1, source)
are kept. Every retriever's score and rank are carried into the model as features. TF-IDF is
fit on each split's own records, without labels, so French tokens get French IDF values. Top-k
search uses `sparse_dot_topn`, a multithreaded sparse top-k matrix product, processed in chunks of S1 rows.

`01_blocking_train.json` reports: recall at caps 1…60, recall per retriever and per source, **the
number of true pairs found only by each retriever** (to drop dead retrievers), cross-country
misses, the **macro-F0.5 ceiling** (what a perfect matcher would score on these candidates), and
25 missed-pair samples.

| | |
|---|---|
| Why it helps | Recall lost here can't be recovered later. Each retriever covers a different kind of noise, and splitting by source prevents S2 from crowding out S3. |
| Precision | Neutral (the model decides), but retrieval ranks are strong features |
| Recall | Target ≥99% of true pairs within the cap |
| F0.5 | The ceiling is set here: a missed pair on an entity with one true match turns that entity's score to 0 |
| Cost | O(S1 × k) per retriever, well under an hour on 8 cores for about 1.7M records; memory is bounded by the chunk size |
| Risks | Cross-country true pairs are never retrieved. **Mitigation:** the profile measures them, and a global retriever is added if they're above ~0.2%. Chains overflow key blocks. **Mitigation:** block-size cap; TF-IDF still retrieves them. Too large a cap adds cost. **Mitigation:** the recall-vs-cap table sets it. |

**Planned (gated on the missed-pair samples):** a learned pre-filter (a small LightGBM on retrieval
features only) that allows a wider retrieval (top-50) while keeping the pairs fed to the full model at about 10
per source, and a multilingual dense retriever (FAISS HNSW) if the misses are semantic
(`Chemist` ↔ `Pharmacy`).

## 3. Name matching features — `features.py` (implemented)

Levenshtein distance and normalized similarity, Jaro-Winkler, ratio, partial ratio, token-sort, token-set
(all rapidfuzz `cpdist`, vectorized in C), token Jaccard, char-3-gram Jaccard, **IDF-weighted
Jaccard and min-coverage**, **IDF mass of unmatched tokens, and its maximum** (the `Roasters` feature),
containment, first/last-token equality, acronym match, number-in-name agreement (`Store #12` vs
`#14`), legal-form agreement, Metaphone of the first token, token counts, and name frequency among S1 and all
records (how chain-like the name is).

* **Precision:** +++. The unmatched-IDF, number-conflict and frequency features target the most common
  false merges: franchises and branches, and names with an extra distinguishing word.
* **Recall:** +. Token-set and partial ratios survive truncation and reordering.
* **Cost:** about 1–2 µs per scorer per pair in C; the set features run in a process pool.
* **Risk:** a TF-IDF cosine feature would duplicate the retrieval scores, which are already included.

## 4. Address matching features (implemented)

Same string scorers on the normalized address, IDF-weighted overlap and unmatched-address mass,
**postcode equal / conflict / unknown** (three-valued, because a missing postcode must not be treated as a mismatch), same
first 3 postcode digits, **house number equal/conflict**, numeric-token Jaccard and symmetric
difference (unit and suite numbers), landmark overlap, city/state overlap (address tail), postcode density
(how many businesses share the postcode), and missing-field flags.

* **Precision:** +++. A conflicting house or postcode is the strongest evidence against branches of a chain.
* **Recall:** +. Three-valued comparisons avoid penalizing missing fields.
* **Risk:** postcode extraction on French and Indian formats. **Mitigation:** the profile reports postcode coverage per
  source, and extraction is checked on the printed France samples.

## 5. Country-aware matching

Country only defines the blocking partition, the IDF statistics and the frequency features, all of which are computed
per split, so France gets its own. There is **no country one-hot and no per-country model or threshold**. Leave-one-country-out
validation (§11) estimates the score on an unseen country, and features that help only within one country are dropped.

## 6. Relational ("graph") features (implemented) — the main precision lever

For each pair and several signals (`pair_sim`, name token-set, address token-set, name IDF-Jaccard, RRF):
* the pair's rank within its S1's candidate list, and its **margin over the best competing candidate**;
* the S1's rank among all S1 entities that retrieved the same candidate, and its **margin over the best competing S1**;
* candidate-list size, and the number of S1 entities competing for the candidate.

Source 1 is deduplicated, so if a candidate matches S1-A clearly better than S1-B, it isn't a match
for S1-B. These features encode that directly.
**Exclusivity at decode time** (each S2/S3 record goes to at most its best S1) is tried on and
off during decoding. The profile reports whether the ground truth ever maps one S2/S3 id to two S1 entities.

**Planned:** S2↔S3 evidence. If an S3 record strongly matches an S2 record that matches S1-A, that raises
the probability of S3→S1-A. This adds second-pass features from first-pass out-of-fold probabilities
(a max over siblings of sim(S3, S2′) × p(S1, S2′)).

| Precision | +++ | Recall | + (S2↔S3 evidence brings back weak S3 matches) | Cost | groupby operations only, a few minutes | Risk | leakage if these features are built from in-fold predictions. **Mitigation:** only out-of-fold predictions are used. |

## 7. Embeddings (planned, gated)

Sentence embeddings are added only if `01_blocking_train.json` and `03_train.json` show semantic
misses. The model would be **multilingual-e5-small** (MIT, 118M parameters, inside the license and size rules), used for a dense
retriever (FAISS HNSW) and a cosine feature. On CPU, roughly 2–4 h for all records. On a Kaggle GPU, about 15 min.

## 8. Hard negatives (implemented by construction)

Training pairs are the blocking output: all positives plus every retrieved non-match, up to 25 per source for each S1.
That set already contains the `Star Coffee Roasters` cases: same name different branch, same address
different tenant, and name containment. It also keeps training and inference distributions identical.
Errors are reviewed in the top false-positive samples in `03_train.json`. If one false-positive family dominates, a
targeted feature is added for it. Random extra negatives are not added.

## 9. Models & ensemble — `train.py` (implemented)

* LightGBM (lr 0.03, 127 leaves, early stopping), XGBoost (hist), CatBoost (depth 8).
  5-fold **GroupKFold by S1 id**, so all candidates of an entity stay in the same fold.
* The stacker is logistic regression on base-model logits, evaluated out-of-fold with the same folds, and used only if it
  beats the best single model on macro F0.5.
* Reported: AUC, AP and log-loss per model; gain importances; zero-gain features.

| Precision/Recall | typically +0.3–1.0 F0.5 points over LightGBM alone | Cost | CatBoost dominates on CPU; disable it for fast iterations | Risk | Overfitting the stacker. **Mitigation:** logistic regression only, with a handful of inputs. |

**Planned deep model:** fine-tuning a multilingual cross-encoder (e.g. multilingual-MiniLM-L12,
Apache-2.0) on the uncertain pairs only (0.05 < p < 0.95), with its score added as a stacker input. This
runs on a GPU (Kaggle) and is worth doing once the GBDT stops improving.

## 10. Decoding & threshold optimization — `decode.py` (implemented)

For each S1 entity, candidates are sorted by probability. For each prefix size k, the exact expected F0.5 is computed
under independence, using Poisson-binomial distributions of the true positives inside and outside the prefix,
and compared with P(no match) for the empty set. The largest value wins. The out-of-fold grid covers:
{expected-F0.5 with logit bias −1…+1, global threshold 0.30…0.95} × {exclusivity on, off}, scored
with the **exact** leaderboard metric over **all** train S1 entities, including those whose true matches blocking missed.

| Precision | ++ on singletons | Recall | + on multi-match entities | F0.5 | typically +1–2 points over a single global threshold | Cost | seconds to a few minutes | Risk | Calibration drift on France. **Mitigation:** the chosen bias is checked on leave-one-country-out, and `04_predict.json` compares probability quantiles and match rates per country. |

## 11. Validation strategy — overfitting and leaderboard stability

* 5-fold GroupKFold by S1 for all model and decoding choices. Out-of-fold macro F0.5 is the main score.
* **Leave-one-country-out**: train on India, score the US, and the reverse. The gap between this and the in-country
  score shows how much we're overfitting to specific countries. A change is accepted only if it doesn't hurt this score.
* The public leaderboard is a subset of the test set, so decisions are driven by the cross-validation score. A leaderboard gain without a
  cross-validation gain is treated as noise.
* Few hyperparameters are tuned (GBDT defaults plus early stopping). Most gains should come from recall and features,
  not from tuning.

## 12. Error-analysis loop

Every training run reports the score lost by bucket (false merges on singletons, entities where all matches were missed,
partial matches), F0.5 by number of true matches, per country, the top-20 false positives and false negatives with text, and blocking-miss samples.
Each iteration targets the largest bucket:

1. Blocking misses → new keys or retrievers, or a higher `k`
2. False merges on singletons → frequency, conflict and competition features; the decoding bias
3. Partial matches → S2↔S3 evidence, exclusivity
4. Leave-one-country-out gap → remove features that only help within one country, and check normalization on the France samples

## 13. Production-scale inference

Everything is chunked and streamed: blocking is written as Parquet per partition, features use
multithreaded C scorers plus a process pool, and models are averaged across folds. Rough cost for about 1.7M test records
on an 8-core, 16–32 GB machine: normalization ~5 min, blocking ~15–40 min, features ~20–40 min,
scoring ~5 min. `candidate_pairs.tsv` is exactly the set the model scores, as the rules require,
and the official validator runs at the end.

## 14. Roadmap in expected-gain order (decided after each report)

1. v1 baseline → read the profile, blocking and training reports
2. Blocking fixes from the missed-pair samples (probably the largest single gain)
3. Features targeting the top false-positive family
4. S2↔S3 evidence (second pass)
5. Learned pre-filter plus a wider retrieval
6. Cross-encoder on uncertain pairs (GPU)
7. Final: full-data retraining, re-tuning the decoding on out-of-fold data, validator, submission zip
