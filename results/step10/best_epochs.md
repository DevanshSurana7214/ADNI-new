# Step 10 - best-epoch distribution for every new run

Neural runs only (B1 has no epochs). `best_epoch` is the epoch selected by each checkpoint rule; a run is flagged when any of its epochs is 0.

| source | arm | ckpt rule | runs | median | mean | min | max | #(==0) | flag epoch 0 |
|---|---|---|---|---|---|---|---|---|---|
| S1 | A0 | primary | 15 | 3.0 | 4.00 | 0 | 12 | 1 | YES |
| S1 | A1 | primary | 15 | 7.0 | 9.33 | 0 | 35 | 1 | YES |
| S1 | A4 | primary | 15 | 10.0 | 8.87 | 0 | 23 | 1 | YES |
| S1 | C1 | primary | 15 | 9.0 | 11.47 | 0 | 38 | 1 | YES |
| S1 | Cshuf | primary | 15 | 3.0 | 3.67 | 0 | 12 | 1 | YES |
| S2 | A0 | primary | 3 | 3.0 | 4.33 | 1 | 9 | 0 | no |
| S2 | A1 | primary | 3 | 4.0 | 4.00 | 1 | 7 | 0 | no |
| S2 | A2 | primary | 3 | 4.0 | 5.00 | 2 | 9 | 0 | no |
| S2 | A3 | primary | 3 | 7.0 | 10.33 | 7 | 17 | 0 | no |
| S2 | A4 | primary | 3 | 7.0 | 6.00 | 4 | 7 | 0 | no |
| S1 | A0 | secondary | 15 | 5.0 | 6.00 | 1 | 17 | 0 | no |
| S1 | A1 | secondary | 15 | 3.0 | 5.20 | 0 | 23 | 2 | YES |
| S1 | A4 | secondary | 15 | 5.0 | 4.53 | 0 | 13 | 4 | YES |
| S1 | C1 | secondary | 15 | 4.0 | 6.13 | 0 | 31 | 3 | YES |
| S1 | Cshuf | secondary | 15 | 4.0 | 4.73 | 0 | 12 | 1 | YES |
| S2 | A0 | secondary | 3 | 3.0 | 4.00 | 1 | 8 | 0 | no |
| S2 | A1 | secondary | 3 | 3.0 | 2.67 | 1 | 4 | 0 | no |
| S2 | A2 | secondary | 3 | 3.0 | 4.33 | 2 | 8 | 0 | no |
| S2 | A3 | secondary | 3 | 3.0 | 2.67 | 1 | 4 | 0 | no |
| S2 | A4 | secondary | 3 | 1.0 | 2.00 | 1 | 4 | 0 | no |
