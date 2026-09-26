# Experiment log

Auto-generated from `artifacts/experiments.jsonl` by `src.evaluate.render_log`.

| time | name | f05 | macro_p | macro_r | singleton_acc | pair_recall | oracle_f05 | cands_per_s1 | notes |
|---|---|---|---|---|---|---|---|---|---|
| 2026-09-25 01:31 | baseline_all_empty | 0.0559 |  | 0.0000 | 1.0000 |  |  |  | predict no matches for everyone |
| 2026-09-25 01:31 | baseline_all_empty | 0.0545 |  | 0.0000 | 1.0000 |  |  |  | predict no matches for everyone [dev 20k] |
| 2026-09-25 01:31 | block_exact_namekey |  |  |  |  | 0.6561 | 0.8514 | 63.2882 | candidates = exact name_key within country [dev 20k] |
| 2026-09-25 01:31 | rule_namekey_all | 0.4674 | 0.5037 | 0.6587 | 0.2936 |  |  |  | predict every exact name_key candidate [dev 20k] |
| 2026-09-25 01:31 | rule_namekey_state | 0.6038 | 0.6652 | 0.6560 | 0.4156 |  |  |  | exact name_key + same state (or missing) [dev 20k] |
| 2026-09-25 01:31 | rule_namekey_state_house | 0.6865 | 0.9826 | 0.4750 | 0.9633 |  |  |  | exact name_key + state + same house number [dev 20k] |
