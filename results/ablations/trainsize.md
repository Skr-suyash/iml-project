# Training-set size

Fixed 70% test split; train on a random subset of the 30% train split. Area lift (accuracy points), mean ± std over 5 splits.

## Cost ratio 20

| n train | Ours: greedy VOI | Fixed V1 (answer confidence) | Fixed V2 (self-consistency, best k) | Per-category map (oracle task) | Pre-generation router (V0) | AutoMix (V4, Bayes/POMDP) | Compute everything |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 50 | 1.53 ± 0.61 | 2.79 ± 0.07 | 1.96 ± 0.09 | 2.02 ± 0.49 | 1.52 ± 0.31 | 0.31 ± 0.18 | -0.00 ± 0.00 |
| 100 | 2.43 ± 0.65 | 2.79 ± 0.07 | 1.96 ± 0.09 | 1.82 ± 0.36 | 1.80 ± 0.39 | 0.44 ± 0.04 | -0.00 ± 0.00 |
| 200 | 3.10 ± 0.62 | 2.79 ± 0.07 | 1.96 ± 0.09 | 1.87 ± 0.25 | 1.92 ± 0.33 | 0.37 ± 0.12 | -0.00 ± 0.00 |
| 400 | 3.35 ± 0.25 | 2.79 ± 0.07 | 1.96 ± 0.09 | 1.88 ± 0.20 | 1.93 ± 0.14 | 0.40 ± 0.10 | -0.00 ± 0.00 |
| all | 3.85 ± 0.26 | 2.79 ± 0.07 | 1.96 ± 0.09 | 1.87 ± 0.18 | 2.32 ± 0.18 | 0.44 ± 0.09 | -0.00 ± 0.00 |

## Cost ratio 60

| n train | Ours: greedy VOI | Fixed V1 (answer confidence) | Fixed V2 (self-consistency, best k) | Per-category map (oracle task) | Pre-generation router (V0) | AutoMix (V4, Bayes/POMDP) | Compute everything |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 50 | 2.57 ± 0.48 | 3.06 ± 0.07 | 3.05 ± 0.15 | 3.45 ± 0.35 | 1.52 ± 0.31 | 0.78 ± 0.38 | 1.02 ± 0.40 |
| 100 | 3.23 ± 0.56 | 3.06 ± 0.07 | 3.05 ± 0.15 | 3.14 ± 0.39 | 1.80 ± 0.39 | 1.03 ± 0.08 | 1.56 ± 0.27 |
| 200 | 3.72 ± 0.57 | 3.06 ± 0.07 | 3.05 ± 0.15 | 3.09 ± 0.22 | 1.92 ± 0.33 | 1.00 ± 0.14 | 1.85 ± 0.39 |
| 400 | 3.96 ± 0.25 | 3.06 ± 0.07 | 3.05 ± 0.15 | 3.20 ± 0.35 | 1.93 ± 0.14 | 1.07 ± 0.14 | 2.04 ± 0.27 |
| all | 4.47 ± 0.26 | 3.06 ± 0.07 | 3.05 ± 0.15 | 3.14 ± 0.20 | 2.32 ± 0.18 | 1.15 ± 0.12 | 2.40 ± 0.20 |

