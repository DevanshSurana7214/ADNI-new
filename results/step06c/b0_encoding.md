# Step 6c - B0 encoding reconciliation

| B0 encoding | features | macro-AUROC (uniform CI) |
|-------------|----------|---------------------------|
| b0_lin | 4 | 0.6001 [0.5883, 0.6123] |
| b0_lin_inter | 10 | 0.6174 [0.6063, 0.6289] |
| b0_oh | 15 | 0.6161 [0.6049, 0.6275] |

Gap one-hot minus four-bit = +0.0160.
Gap one-hot minus (four-bit + pairwise interactions) = +0.0013.

If the four-bit-plus-interactions model recovers the one-hot macro-AUROC, the encoding gap is representational capacity (the 15-cell model can express the interaction structure between availability bits), not L2 shrinkage on the four-bit model.
