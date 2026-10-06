# Step 10 - retrain verification (A0/A1/A4 reproduce the Step 9 OOF predictions)

Max absolute deviation of the retrained test-fold probabilities from the Step 9 `oof_all.parquet`, matched on (episode_id, seed, fold, arm, ckpt_rule, operating_point).

| arm | ckpt rule | rows | max |dp| |
|---|---|---|---|
| A0 | primary | 39720 | 0.000e+00 |
| A0 | secondary | 39720 | 0.000e+00 |
| A1 | primary | 79440 | 0.000e+00 |
| A1 | secondary | 79440 | 0.000e+00 |
| A4 | primary | 79440 | 0.000e+00 |
| A4 | secondary | 79440 | 0.000e+00 |

**Global max deviation = 0.000e+00** - training is deterministic and the retrain reproduces the Step 9 predictions exactly.
