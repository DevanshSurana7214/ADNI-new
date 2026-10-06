# Step 11a - best-epoch distribution per run-set and checkpoint rule

`best_epoch` is the epoch selected by each rule. A run-set is flagged when its median best epoch is below 5, or when ANY of its runs selected epoch 0.

| run-set | family | rule | runs | median | mean | min | max | #(<5) | flag median<5 | #(==0) | flag epoch 0 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| R1_A0 | R1 | primary | 15 | 3.0 | 4.00 | 0 | 12 | 9 | YES | 1 | YES |
| R1_A1 | R1 | primary | 15 | 7.0 | 9.33 | 0 | 35 | 5 | no | 1 | YES |
| R1_A2 | R1 | primary | 15 | 3.0 | 3.87 | 0 | 16 | 10 | YES | 1 | YES |
| R1_A3 | R1 | primary | 15 | 10.0 | 12.93 | 0 | 56 | 3 | no | 1 | YES |
| R1_A4 | R1 | primary | 15 | 10.0 | 8.87 | 0 | 23 | 5 | no | 1 | YES |
| R2_A0 | R2 | primary | 15 | 13.0 | 14.80 | 2 | 35 | 2 | no | 0 | no |
| R2_A1 | R2 | primary | 15 | 25.0 | 25.67 | 9 | 45 | 0 | no | 0 | no |
| R2_A4 | R2 | primary | 15 | 23.0 | 33.80 | 7 | 82 | 0 | no | 0 | no |
| A4_tau0p5 | sensitivity | primary | 15 | 4.0 | 6.27 | 0 | 19 | 8 | YES | 2 | YES |
| A4_tau1p5 | sensitivity | primary | 15 | 13.0 | 18.87 | 4 | 56 | 2 | no | 0 | no |
| A3_clip0p5_2 | sensitivity | primary | 15 | 10.0 | 8.40 | 0 | 21 | 4 | no | 1 | YES |
| A3_clip0p1_10 | sensitivity | primary | 15 | 10.0 | 10.20 | 0 | 33 | 3 | no | 1 | YES |
| A0_pdrop0 | sensitivity | primary | 15 | 2.0 | 4.33 | 0 | 24 | 11 | YES | 1 | YES |
| A0_pdrop0p5 | sensitivity | primary | 15 | 3.0 | 4.33 | 0 | 10 | 9 | YES | 2 | YES |
| A4_pdrop0 | sensitivity | primary | 15 | 7.0 | 10.40 | 0 | 33 | 5 | no | 2 | YES |
| A4_pdrop0p5 | sensitivity | primary | 15 | 8.0 | 10.87 | 0 | 40 | 3 | no | 1 | YES |
| R1_A0 | R1 | secondary | 15 | 5.0 | 6.00 | 1 | 17 | 6 | no | 0 | no |
| R1_A1 | R1 | secondary | 15 | 3.0 | 5.20 | 0 | 23 | 11 | YES | 2 | YES |
| R1_A2 | R1 | secondary | 15 | 3.0 | 4.60 | 1 | 16 | 8 | YES | 0 | no |
| R1_A3 | R1 | secondary | 15 | 4.0 | 3.87 | 0 | 15 | 9 | YES | 4 | YES |
| R1_A4 | R1 | secondary | 15 | 5.0 | 4.53 | 0 | 13 | 7 | no | 4 | YES |
| R2_A0 | R2 | secondary | 15 | 10.0 | 15.93 | 2 | 39 | 5 | no | 0 | no |
| R2_A1 | R2 | secondary | 15 | 21.0 | 21.47 | 4 | 59 | 1 | no | 0 | no |
| R2_A4 | R2 | secondary | 15 | 15.0 | 22.67 | 3 | 82 | 2 | no | 0 | no |
| A4_tau0p5 | sensitivity | secondary | 15 | 4.0 | 4.07 | 0 | 17 | 11 | YES | 3 | YES |
| A4_tau1p5 | sensitivity | secondary | 15 | 5.0 | 9.47 | 0 | 42 | 7 | no | 1 | YES |
| A3_clip0p5_2 | sensitivity | secondary | 15 | 3.0 | 2.87 | 0 | 7 | 10 | YES | 5 | YES |
| A3_clip0p1_10 | sensitivity | secondary | 15 | 2.0 | 3.93 | 0 | 15 | 8 | YES | 4 | YES |
| A0_pdrop0 | sensitivity | secondary | 15 | 4.0 | 3.80 | 0 | 10 | 11 | YES | 1 | YES |
| A0_pdrop0p5 | sensitivity | secondary | 15 | 3.0 | 4.93 | 0 | 15 | 10 | YES | 2 | YES |
| A4_pdrop0 | sensitivity | secondary | 15 | 2.0 | 2.73 | 0 | 7 | 11 | YES | 3 | YES |
| A4_pdrop0p5 | sensitivity | secondary | 15 | 3.0 | 6.27 | 0 | 40 | 10 | YES | 4 | YES |
