# Results

## Cost ratio LLM/SLM = 5

| Method | IBC lift (%) | Area lift (acc. points) | Max accuracy |
| --- | --- | --- | --- |
| Fixed V1 (answer confidence) | 26.3 ± 17.6 | 2.39 ± 0.93 | 0.680 |
| Ours: greedy VOI | 37.1 ± 16.1 | 1.81 ± 1.15 | 0.668 |
| Pre-generation router (V0) | 12.1 ± 14.9 | 0.69 ± 0.66 | 0.672 |
| Fixed V2 (self-consistency, best k) | -11.4 ± 9.4 | -1.30 ± 0.94 | 0.690 |
| Fixed V3 (retrieval verify) | -22.9 ± 12.2 | -2.36 ± 1.35 | 0.672 |
| AutoMix (V4, Bayes/POMDP) | -37.3 ± 12.2 | -3.43 ± 1.55 | 0.674 |
| Fixed V4 (context-free verify) | -44.9 ± 10.1 | -4.37 ± 1.31 | 0.672 |
| Per-category map (oracle task) | -61.8 ± 9.9 | -5.54 ± 1.48 | 0.674 |
| Compute everything | -98.0 ± 0.4 | -9.41 ± 1.74 | 0.688 |

Always-SLM: cost 1.00, acc 0.480. Always-LLM: cost 5.00, acc 0.690 (split 0).

## Cost ratio LLM/SLM = 20

| Method | IBC lift (%) | Area lift (acc. points) | Max accuracy |
| --- | --- | --- | --- |
| Fixed V1 (answer confidence) | 44.9 ± 19.9 | 3.64 ± 0.93 | 0.680 |
| Ours: greedy VOI | 57.9 ± 18.7 | 2.81 ± 0.94 | 0.670 |
| Fixed V2 (self-consistency, best k) | 26.9 ± 24.7 | 2.14 ± 1.08 | 0.690 |
| Fixed V3 (retrieval verify) | 20.4 ± 17.3 | 1.22 ± 0.95 | 0.672 |
| Pre-generation router (V0) | 12.1 ± 14.9 | 0.69 ± 0.66 | 0.672 |
| Per-category map (oracle task) | 4.0 ± 6.3 | 0.31 ± 0.79 | 0.674 |
| AutoMix (V4, Bayes/POMDP) | 8.7 ± 15.2 | 0.28 ± 1.08 | 0.674 |
| Fixed V4 (context-free verify) | -4.2 ± 13.5 | -0.87 ± 1.02 | 0.672 |
| Compute everything | -65.9 ± 6.9 | -5.04 ± 1.56 | 0.688 |

Always-SLM: cost 1.00, acc 0.480. Always-LLM: cost 20.00, acc 0.690 (split 0).

## Cost ratio LLM/SLM = 60

| Method | IBC lift (%) | Area lift (acc. points) | Max accuracy |
| --- | --- | --- | --- |
| Fixed V1 (answer confidence) | 48.6 ± 20.3 | 3.87 ± 0.94 | 0.680 |
| Fixed V2 (self-consistency, best k) | 52.5 ± 25.2 | 3.80 ± 0.96 | 0.690 |
| Ours: greedy VOI | 68.4 ± 24.1 | 3.70 ± 1.00 | 0.670 |
| Per-category map (oracle task) | 55.1 ± 25.9 | 2.51 ± 1.41 | 0.674 |
| Fixed V3 (retrieval verify) | 32.5 ± 18.1 | 2.04 ± 0.82 | 0.672 |
| Compute everything | 8.7 ± 18.1 | 1.82 ± 1.17 | 0.688 |
| AutoMix (V4, Bayes/POMDP) | 22.3 ± 16.1 | 1.14 ± 0.99 | 0.674 |
| Pre-generation router (V0) | 12.1 ± 14.9 | 0.69 ± 0.66 | 0.672 |
| Fixed V4 (context-free verify) | 7.9 ± 13.9 | -0.02 ± 0.98 | 0.672 |

Always-SLM: cost 1.00, acc 0.480. Always-LLM: cost 60.00, acc 0.690 (split 0).
