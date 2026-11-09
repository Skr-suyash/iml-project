# Paired bootstrap: ours (greedy VOI) minus each baseline

1000 resamples of the test queries per split, averaged over 5 splits. Mean difference with 95% interval; p = share of resamples where ours is not better.

| Cost ratio | Baseline | Area lift diff (points) | p | IBC lift diff (%) | p |
| --- | --- | --- | --- | --- | --- |
| 3 | Fixed V1 (answer confidence) | 2.13 [1.80, 2.44] | 0.000 | 35.9 [31.3, 40.8] | 0.000 |
| 3 | Fixed V2 (self-consistency, best k) | 2.40 [2.13, 2.68] | 0.000 | 39.8 [35.5, 44.7] | 0.000 |
| 3 | Fixed V3 (retrieval verify) | 2.40 [2.13, 2.68] | 0.000 | 39.8 [35.5, 44.7] | 0.000 |
| 3 | Fixed V4 (context-free verify) | 2.40 [2.13, 2.68] | 0.000 | 39.8 [35.5, 44.7] | 0.000 |
| 3 | Per-category map (oracle task) | 2.40 [2.13, 2.68] | 0.000 | 39.8 [35.5, 44.7] | 0.000 |
| 3 | Pre-generation router (V0) | 0.01 [-0.15, 0.16] | 0.454 | 5.7 [2.9, 8.5] | 0.000 |
| 3 | AutoMix (V4, Bayes/POMDP) | 2.40 [2.13, 2.68] | 0.000 | 39.8 [35.5, 44.7] | 0.000 |
| 3 | Compute everything | 2.40 [2.13, 2.68] | 0.000 | 39.8 [35.5, 44.7] | 0.000 |
| 5 | Fixed V1 (answer confidence) | 1.33 [1.02, 1.62] | 0.000 | 30.5 [25.9, 34.9] | 0.000 |
| 5 | Fixed V2 (self-consistency, best k) | 2.78 [2.50, 3.06] | 0.000 | 48.8 [44.2, 53.9] | 0.000 |
| 5 | Fixed V3 (retrieval verify) | 2.78 [2.50, 3.06] | 0.000 | 48.8 [44.2, 53.9] | 0.000 |
| 5 | Fixed V4 (context-free verify) | 2.78 [2.50, 3.06] | 0.000 | 48.8 [44.2, 53.9] | 0.000 |
| 5 | Per-category map (oracle task) | 2.78 [2.50, 3.06] | 0.000 | 48.8 [44.2, 53.9] | 0.000 |
| 5 | Pre-generation router (V0) | 0.38 [0.18, 0.60] | 0.000 | 14.7 [11.0, 18.3] | 0.000 |
| 5 | AutoMix (V4, Bayes/POMDP) | 2.78 [2.50, 3.06] | 0.000 | 48.8 [44.2, 53.9] | 0.000 |
| 5 | Compute everything | 2.78 [2.50, 3.06] | 0.000 | 48.8 [44.2, 53.9] | 0.000 |
| 10 | Fixed V1 (answer confidence) | 0.95 [0.64, 1.25] | 0.000 | 27.9 [23.2, 32.8] | 0.000 |
| 10 | Fixed V2 (self-consistency, best k) | 2.39 [2.11, 2.69] | 0.000 | 46.3 [41.6, 50.8] | 0.000 |
| 10 | Fixed V3 (retrieval verify) | 3.35 [3.07, 3.63] | 0.000 | 57.3 [52.3, 62.2] | 0.000 |
| 10 | Fixed V4 (context-free verify) | 3.34 [3.06, 3.64] | 0.000 | 57.3 [52.2, 62.1] | 0.000 |
| 10 | Per-category map (oracle task) | 2.79 [2.48, 3.13] | 0.000 | 51.4 [46.7, 56.5] | 0.000 |
| 10 | Pre-generation router (V0) | 0.96 [0.73, 1.19] | 0.000 | 23.3 [19.1, 27.4] | 0.000 |
| 10 | AutoMix (V4, Bayes/POMDP) | 3.35 [3.06, 3.65] | 0.000 | 57.3 [52.3, 62.2] | 0.000 |
| 10 | Compute everything | 3.36 [3.08, 3.65] | 0.000 | 57.5 [52.4, 62.4] | 0.000 |
| 20 | Fixed V1 (answer confidence) | 1.04 [0.77, 1.30] | 0.000 | 28.8 [24.2, 33.3] | 0.000 |
| 20 | Fixed V2 (self-consistency, best k) | 1.89 [1.61, 2.20] | 0.000 | 39.5 [35.0, 43.9] | 0.000 |
| 20 | Fixed V3 (retrieval verify) | 2.92 [2.56, 3.29] | 0.000 | 53.0 [47.6, 58.0] | 0.000 |
| 20 | Fixed V4 (context-free verify) | 3.25 [2.90, 3.59] | 0.000 | 55.4 [50.2, 60.7] | 0.000 |
| 20 | Per-category map (oracle task) | 1.99 [1.65, 2.33] | 0.000 | 42.9 [37.9, 47.7] | 0.000 |
| 20 | Pre-generation router (V0) | 1.48 [1.21, 1.74] | 0.000 | 29.6 [25.4, 33.9] | 0.000 |
| 20 | AutoMix (V4, Bayes/POMDP) | 3.35 [3.01, 3.67] | 0.000 | 56.4 [51.2, 61.7] | 0.000 |
| 20 | Compute everything | 3.88 [3.60, 4.16] | 0.000 | 63.8 [59.1, 68.8] | 0.000 |
| 60 | Fixed V1 (answer confidence) | 1.39 [1.12, 1.66] | 0.000 | 34.0 [29.1, 38.9] | 0.000 |
| 60 | Fixed V2 (self-consistency, best k) | 1.41 [1.15, 1.67] | 0.000 | 32.7 [28.1, 37.2] | 0.000 |
| 60 | Fixed V3 (retrieval verify) | 2.61 [2.27, 2.95] | 0.000 | 50.6 [45.5, 55.4] | 0.000 |
| 60 | Fixed V4 (context-free verify) | 3.18 [2.83, 3.54] | 0.000 | 53.4 [48.1, 59.3] | 0.000 |
| 60 | Per-category map (oracle task) | 1.32 [1.01, 1.61] | 0.000 | 32.8 [28.1, 37.4] | 0.000 |
| 60 | Pre-generation router (V0) | 2.11 [1.83, 2.38] | 0.000 | 38.3 [34.0, 42.9] | 0.000 |
| 60 | AutoMix (V4, Bayes/POMDP) | 3.33 [2.99, 3.67] | 0.000 | 54.8 [49.3, 60.8] | 0.000 |
| 60 | Compute everything | 2.07 [1.90, 2.24] | 0.000 | 44.4 [41.5, 47.2] | 0.000 |
| 200 | Fixed V1 (answer confidence) | 1.71 [1.46, 1.98] | 0.000 | 40.9 [36.3, 45.4] | 0.000 |
| 200 | Fixed V2 (self-consistency, best k) | 1.08 [0.84, 1.31] | 0.000 | 29.8 [25.5, 34.4] | 0.000 |
| 200 | Fixed V3 (retrieval verify) | 2.69 [2.37, 3.03] | 0.000 | 54.4 [49.3, 59.7] | 0.000 |
| 200 | Fixed V4 (context-free verify) | 3.34 [3.00, 3.69] | 0.000 | 57.2 [51.8, 63.5] | 0.000 |
| 200 | Per-category map (oracle task) | 0.87 [0.59, 1.14] | 0.000 | 20.7 [16.2, 25.4] | 0.000 |
| 200 | Pre-generation router (V0) | 2.52 [2.27, 2.81] | 0.000 | 46.5 [42.0, 51.3] | 0.000 |
| 200 | AutoMix (V4, Bayes/POMDP) | 3.50 [3.18, 3.84] | 0.000 | 58.7 [53.3, 64.8] | 0.000 |
| 200 | Compute everything | 0.41 [0.31, 0.51] | 0.000 | 18.2 [16.2, 20.1] | 0.000 |
