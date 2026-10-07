# Step 11b - mask-intervention diagnostic for A5-A8 (EXPLORATORY, primary rule)

EXPLORATORY: no selection among A5-A8; these results must not be used to change A0-A4, the recipes, or any pre-declared rule.

Step 10 protocol: (a) REMOVE an observed modality (mask -> 0); (b) CONTENT SWAP its values with a random other test episode that has it (mask kept). ratio = (a)/(b) > 1 means the model reacts to the mask beyond the lost content.

| arm | modality | n | mean TV remove | mean TV swap | ratio (a)/(b) |
|---|---|---|---|---|---|
| A5 | amy | 12951 | 0.1200 | 0.2136 | 0.562 |
| A5 | fdg | 14763 | 0.0735 | 0.1244 | 0.590 |
| A5 | mri | 22068 | 0.1945 | 0.3512 | 0.554 |
| A5 | tau | 8082 | 0.1495 | 0.2676 | 0.559 |
| A6 | amy | 12951 | 0.1160 | 0.2073 | 0.559 |
| A6 | fdg | 14763 | 0.0734 | 0.1250 | 0.587 |
| A6 | mri | 22068 | 0.1891 | 0.3425 | 0.552 |
| A6 | tau | 8082 | 0.1479 | 0.2674 | 0.553 |
| A7 | amy | 12951 | 0.1210 | 0.2106 | 0.574 |
| A7 | fdg | 14763 | 0.0736 | 0.1223 | 0.601 |
| A7 | mri | 22068 | 0.1984 | 0.3509 | 0.565 |
| A7 | tau | 8082 | 0.1597 | 0.2832 | 0.564 |
| A8 | amy | 12951 | 0.1173 | 0.2100 | 0.559 |
| A8 | fdg | 14763 | 0.0733 | 0.1268 | 0.578 |
| A8 | mri | 22068 | 0.1892 | 0.3424 | 0.552 |
| A8 | tau | 8082 | 0.1482 | 0.2669 | 0.555 |
| A5 | ALL | 57864 | 0.1406 | 0.2509 | 0.561 |
| A6 | ALL | 57864 | 0.1375 | 0.2463 | 0.558 |
| A7 | ALL | 57864 | 0.1438 | 0.2517 | 0.571 |
| A8 | ALL | 57864 | 0.1378 | 0.2472 | 0.557 |

Reference rows (Step 10, recipe R1, same S1 grid and same rho-uniform protocol). Step 10 has no intervention for the R2 arms, so these are context only: they show the same sub-1 pooled ratio under the earlier recipe.

| ref arm | ckpt | modality | n | mean TV remove | mean TV swap | ratio (a)/(b) |
|---|---|---|---|---|---|---|
| A4 [Step 10, R1] | primary | ALL | 57864 | 0.1419 | 0.2423 | 0.586 |
| A4 [Step 10, R1] | secondary | ALL | 57864 | 0.1265 | 0.2233 | 0.567 |
