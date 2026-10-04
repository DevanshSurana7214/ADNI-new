# Step 6c - record of the field-strength decision (script-generated)

Source artifacts: ``results/step06b/decision_rule.json``, ``results/step06b/b1_importance.json``, ``results/step06b/b0b_field_strength_colprot_class.csv``.

## Decision

- Rule: if B0b macro-AUROC CI lower bound > 0.55 -> EXCLUDE mri_field_strength from the primary neural models (keep as a sensitivity arm); else keep it.
- B0b macro-AUROC = 0.5412 [0.5273, 0.5555] (threshold 0.55).
- Branch: **KEEP mri_field_strength in the primary models**.

**Fixed decision:** ``mri_field_strength`` is KEPT in the primary neural models. A model trained without it is run only as a sensitivity arm; it is not the primary arm.

## Supporting numbers (Step 6b)

- The four B1 mask bits together carry a gain share of 4.79e-06 (essentially zero).
- ``mri_ST24TA`` is the top B1 feature by gain (share 0.0572); ``mri_mri_field_strength`` ranks 40 of 592 (share 0.00362).

### mri_field_strength x COLPROT x class (MRI-present episodes)

| field_strength | COLPROT | class | n |
|---|---|---|---|
| 0 | ADNI1 | CN | 976 |
| 0 | ADNI1 | MCI | 1407 |
| 0 | ADNI1 | Dementia | 935 |
| 0 | ADNI2 | CN | 158 |
| 0 | ADNI2 | MCI | 125 |
| 0 | ADNI2 | Dementia | 107 |
| 0 | ADNI3 | CN | 0 |
| 0 | ADNI3 | MCI | 0 |
| 0 | ADNI3 | Dementia | 0 |
| 0 | ADNI4 | CN | 0 |
| 0 | ADNI4 | MCI | 0 |
| 0 | ADNI4 | Dementia | 0 |
| 0 | ADNIGO | CN | 67 |
| 0 | ADNIGO | MCI | 49 |
| 0 | ADNIGO | Dementia | 44 |
| 1 | ADNI1 | CN | 83 |
| 1 | ADNI1 | MCI | 96 |
| 1 | ADNI1 | Dementia | 80 |
| 1 | ADNI2 | CN | 1249 |
| 1 | ADNI2 | MCI | 1747 |
| 1 | ADNI2 | Dementia | 688 |
| 1 | ADNI3 | CN | 1460 |
| 1 | ADNI3 | MCI | 1016 |
| 1 | ADNI3 | Dementia | 328 |
| 1 | ADNI4 | CN | 807 |
| 1 | ADNI4 | MCI | 498 |
| 1 | ADNI4 | Dementia | 129 |
| 1 | ADNIGO | CN | 9 |
| 1 | ADNIGO | MCI | 250 |
| 1 | ADNIGO | Dementia | 2 |

field_strength = 0 occurs almost entirely in ADNI1 / ADNI-GO (plus a small ADNI2 remainder) and **never** in ADNI3 / ADNI4, so the binary flag is a strong proxy for acquisition era/protocol; this is why Step 6b treats it as a shortcut-relevant feature and why removing it is a sensitivity analysis, not the primary specification.
