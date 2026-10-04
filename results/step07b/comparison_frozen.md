# Step 7b - A0 pilot, S1 seed 0 fold 0 (frozen config)

Test-fold metrics; the pilot test fold is used here and nowhere else in Step 7b.

| arm | n_ep | macro-AUROC | macro-F1 | BAcc | PN-BAcc (uniform) | PN-BAcc (P(c)) |
|---|---|---|---|---|---|---|
| A0 (old config, model_v0) | 2647 | 0.7866 | 0.6211 | 0.6212 | 0.5924 | 0.5923 |
| A0 (frozen config) | 2647 | 0.7844 | 0.6215 | 0.6327 | 0.6129 | 0.6128 |
| B1 | 2647 | 0.7991 | 0.6311 | 0.6247 | 0.5851 | 0.5853 |
| B1-blind | 2647 | 0.7413 | 0.5751 | 0.5687 | 0.5552 | 0.5551 |
| B2_mri | 2439 | 0.7517 | 0.5849 | 0.5772 | 0.5688 | 0.5687 |

`A0 (old config)` is the deterministic rerun with `configs/model_v0.yaml` (lr 3e-4, wd 0.05, dropout 0.2/0.1); the committed `results/step07` was an unseeded draw of the same config. `B1`, `B1-blind` and `B2_mri` are the Step 6 reference arms on the same fold-0 test episodes.
