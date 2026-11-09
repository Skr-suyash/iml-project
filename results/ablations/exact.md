# Exact planner vs. greedy (H5)

400-query test subsample per split, 3 splits, 12 lambda values.

| Cost ratio | Method | IBC lift (%) | Area lift (acc. points) |
| --- | --- | --- | --- |
| 5 | exact (max 4 samples) | 52.0 ± 14.3 | 2.90 ± 0.51 |
| 5 | greedy (max 4 samples) | 52.2 ± 14.4 | 2.92 ± 0.54 |
| 5 | greedy (max 8 samples) | 52.2 ± 14.4 | 2.92 ± 0.54 |
| 20 | exact (max 4 samples) | 68.9 ± 6.3 | 4.10 ± 0.22 |
| 20 | greedy (max 4 samples) | 67.1 ± 5.5 | 3.89 ± 0.38 |
| 20 | greedy (max 8 samples) | 67.1 ± 5.5 | 3.89 ± 0.38 |
| 60 | exact (max 4 samples) | 77.1 ± 7.2 | 4.57 ± 0.18 |
| 60 | greedy (max 4 samples) | 78.1 ± 6.8 | 4.58 ± 0.35 |
| 60 | greedy (max 8 samples) | 78.3 ± 6.2 | 4.58 ± 0.33 |
