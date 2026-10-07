# Step 11b - embedding-swap test for A5-A8 (EXPLORATORY)

EXPLORATORY: no selection among A5-A8; these results must not be used to change A0-A4, the recipes, or any pre-declared rule.

At test time every episode's pattern embedding is replaced by that of a random OTHER test episode (class-agnostic). TV = total-variation change of the predicted probabilities at the rho-uniform operating point. A large TV on different-pattern swaps means the model relies on the availability pattern beyond the content.

| arm | rule | n | mean TV | median TV | 95th pct TV | same-pattern share | mean TV (diff pattern) |
|---|---|---|---|---|---|---|---|
| A5 | primary | 39720 | 0.0021 | 0.0013 | 0.0068 | 0.209 | 0.0027 |
| A5 | secondary | 39720 | 0.0020 | 0.0014 | 0.0063 | 0.209 | 0.0025 |
| A6 | primary | 39720 | 0.0022 | 0.0013 | 0.0073 | 0.209 | 0.0027 |
| A6 | secondary | 39720 | 0.0021 | 0.0014 | 0.0064 | 0.209 | 0.0026 |
| A7 | primary | 39720 | 0.0023 | 0.0014 | 0.0081 | 0.209 | 0.0029 |
| A7 | secondary | 39720 | 0.0022 | 0.0017 | 0.0064 | 0.209 | 0.0028 |
| A8 | primary | 39720 | 0.0023 | 0.0014 | 0.0077 | 0.209 | 0.0029 |
| A8 | secondary | 39720 | 0.0021 | 0.0015 | 0.0065 | 0.209 | 0.0027 |
