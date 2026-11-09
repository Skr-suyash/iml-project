# Results

## Cost ratio LLM/SLM = 3

| Method | IBC lift (%) | Area lift (acc. points) | Max accuracy |
| --- | --- | --- | --- |
| Ours: greedy VOI | 44.8 ± 4.5 | 3.49 ± 0.13 | 0.786 |
| Pre-generation router (V0) | 45.4 ± 4.6 | 3.45 ± 0.05 | 0.786 |
| Fixed V1 (answer confidence) | 11.9 ± 4.5 | 0.45 ± 0.21 | 0.786 |
| Fixed V2 (self-consistency, best k) | -43.8 ± 1.5 | -4.91 ± 0.16 | 0.786 |
| Fixed V4 (context-free verify) | -76.5 ± 0.8 | -7.78 ± 0.06 | 0.786 |
| Per-category map (oracle task) | -72.4 ± 2.8 | -7.92 ± 0.25 | 0.787 |
| AutoMix (V4, Bayes/POMDP) | -77.4 ± 1.6 | -7.94 ± 0.24 | 0.786 |
| Fixed V3 (retrieval verify) | -78.6 ± 0.9 | -8.74 ± 0.13 | 0.786 |
| Compute everything | -100.0 ± 0.0 | -11.22 ± 0.04 | 0.787 |

Always-SLM: cost 1.00, acc 0.556. Always-LLM: cost 3.00, acc 0.780 (split 0).

## Cost ratio LLM/SLM = 5

| Method | IBC lift (%) | Area lift (acc. points) | Max accuracy |
| --- | --- | --- | --- |
| Ours: greedy VOI | 48.7 ± 4.5 | 3.88 ± 0.11 | 0.786 |
| Pre-generation router (V0) | 45.4 ± 4.6 | 3.45 ± 0.05 | 0.786 |
| Fixed V1 (answer confidence) | 28.4 ± 3.5 | 1.91 ± 0.19 | 0.786 |
| Fixed V2 (self-consistency, best k) | -18.3 ± 2.0 | -2.21 ± 0.20 | 0.786 |
| Per-category map (oracle task) | -37.0 ± 1.7 | -2.99 ± 0.10 | 0.787 |
| Fixed V4 (context-free verify) | -45.2 ± 1.3 | -3.58 ± 0.10 | 0.786 |
| AutoMix (V4, Bayes/POMDP) | -46.6 ± 2.8 | -3.76 ± 0.32 | 0.786 |
| Fixed V3 (retrieval verify) | -49.1 ± 1.4 | -4.53 ± 0.15 | 0.786 |
| Compute everything | -99.9 ± 0.0 | -11.21 ± 0.04 | 0.787 |

Always-SLM: cost 1.00, acc 0.556. Always-LLM: cost 5.00, acc 0.780 (split 0).

## Cost ratio LLM/SLM = 10

| Method | IBC lift (%) | Area lift (acc. points) | Max accuracy |
| --- | --- | --- | --- |
| Ours: greedy VOI | 56.5 ± 2.9 | 4.48 ± 0.04 | 0.786 |
| Pre-generation router (V0) | 45.4 ± 4.6 | 3.45 ± 0.05 | 0.786 |
| Fixed V1 (answer confidence) | 39.2 ± 3.7 | 2.86 ± 0.17 | 0.786 |
| Per-category map (oracle task) | 1.3 ± 2.0 | 1.00 ± 0.11 | 0.787 |
| Fixed V3 (retrieval verify) | -9.7 ± 1.9 | -0.20 ± 0.15 | 0.786 |
| Fixed V2 (self-consistency, best k) | -4.8 ± 1.5 | -0.34 ± 0.11 | 0.786 |
| Fixed V4 (context-free verify) | -10.9 ± 1.6 | -0.35 ± 0.12 | 0.786 |
| AutoMix (V4, Bayes/POMDP) | -12.9 ± 3.8 | -0.51 ± 0.32 | 0.786 |
| Compute everything | -99.8 ± 0.0 | -11.20 ± 0.04 | 0.787 |

Always-SLM: cost 1.00, acc 0.556. Always-LLM: cost 10.00, acc 0.780 (split 0).

## Cost ratio LLM/SLM = 20

| Method | IBC lift (%) | Area lift (acc. points) | Max accuracy |
| --- | --- | --- | --- |
| Ours: greedy VOI | 64.4 ± 2.5 | 4.98 ± 0.10 | 0.786 |
| Pre-generation router (V0) | 45.4 ± 4.6 | 3.45 ± 0.05 | 0.786 |
| Fixed V1 (answer confidence) | 43.6 ± 4.0 | 3.29 ± 0.16 | 0.786 |
| Per-category map (oracle task) | 28.3 ± 2.3 | 3.06 ± 0.07 | 0.787 |
| Fixed V3 (retrieval verify) | 17.0 ± 2.1 | 1.94 ± 0.13 | 0.786 |
| Fixed V4 (context-free verify) | 10.2 ± 2.2 | 1.24 ± 0.13 | 0.786 |
| AutoMix (V4, Bayes/POMDP) | 8.1 ± 4.3 | 1.09 ± 0.32 | 0.786 |
| Fixed V2 (self-consistency, best k) | -3.9 ± 1.0 | 0.97 ± 0.11 | 0.786 |
| Compute everything | -73.2 ± 0.4 | -6.24 ± 0.05 | 0.787 |

Always-SLM: cost 1.00, acc 0.556. Always-LLM: cost 20.00, acc 0.780 (split 0).

## Cost ratio LLM/SLM = 60

| Method | IBC lift (%) | Area lift (acc. points) | Max accuracy |
| --- | --- | --- | --- |
| Ours: greedy VOI | 73.4 ± 1.8 | 5.52 ± 0.09 | 0.786 |
| Per-category map (oracle task) | 48.8 ± 1.9 | 4.50 ± 0.04 | 0.787 |
| Fixed V1 (answer confidence) | 46.9 ± 3.6 | 3.56 ± 0.16 | 0.786 |
| Pre-generation router (V0) | 45.4 ± 4.6 | 3.45 ± 0.05 | 0.786 |
| Fixed V3 (retrieval verify) | 37.6 ± 2.5 | 3.28 ± 0.12 | 0.786 |
| Fixed V2 (self-consistency, best k) | 30.0 ± 1.0 | 2.69 ± 0.05 | 0.786 |
| Fixed V4 (context-free verify) | 29.8 ± 2.1 | 2.26 ± 0.13 | 0.786 |
| Compute everything | -0.1 ± 0.8 | 2.21 ± 0.07 | 0.787 |
| AutoMix (V4, Bayes/POMDP) | 27.0 ± 5.2 | 2.12 ± 0.31 | 0.786 |

Always-SLM: cost 1.00, acc 0.556. Always-LLM: cost 60.00, acc 0.780 (split 0).

## Cost ratio LLM/SLM = 200

| Method | IBC lift (%) | Area lift (acc. points) | Max accuracy |
| --- | --- | --- | --- |
| Ours: greedy VOI | 77.9 ± 3.5 | 5.83 ± 0.12 | 0.786 |
| Per-category map (oracle task) | 61.0 ± 1.6 | 5.15 ± 0.00 | 0.787 |
| Compute everything | 53.1 ± 2.6 | 5.09 ± 0.08 | 0.787 |
| Fixed V2 (self-consistency, best k) | 47.9 ± 1.4 | 3.84 ± 0.11 | 0.786 |
| Fixed V3 (retrieval verify) | 46.4 ± 2.7 | 3.73 ± 0.12 | 0.786 |
| Fixed V1 (answer confidence) | 48.0 ± 3.5 | 3.65 ± 0.15 | 0.786 |
| Pre-generation router (V0) | 45.4 ± 4.6 | 3.45 ± 0.05 | 0.786 |
| Fixed V4 (context-free verify) | 36.6 ± 2.1 | 2.62 ± 0.13 | 0.786 |
| AutoMix (V4, Bayes/POMDP) | 33.4 ± 5.6 | 2.47 ± 0.30 | 0.786 |

Always-SLM: cost 1.00, acc 0.556. Always-LLM: cost 200.00, acc 0.780 (split 0).
