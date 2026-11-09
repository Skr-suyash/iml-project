# Results

## Cost ratio LLM/SLM = 5

| Method | IBC lift (%) | Area lift (acc. points) | Max accuracy |
| --- | --- | --- | --- |
| Fixed V1 (answer confidence) | 32.9 ± 1.2 | 2.42 ± 0.09 | 0.711 |
| Ours: greedy VOI, per-query costs | 38.5 ± 3.6 | 2.39 ± 0.12 | 0.710 |
| Ours: greedy VOI | 34.7 ± 4.8 | 2.01 ± 0.37 | 0.711 |
| Pre-generation router (V0) | 14.0 ± 4.5 | 0.99 ± 0.33 | 0.711 |
| Fixed V2 (self-consistency, best k) | 0.0 ± 0.0 | 0.00 ± 0.00 | 0.710 |
| Fixed V3 (retrieval verify) | 0.0 ± 0.0 | 0.00 ± 0.00 | 0.710 |
| Fixed V4 (context-free verify) | 0.0 ± 0.0 | 0.00 ± 0.00 | 0.710 |
| Per-category map (oracle task) | 0.0 ± 0.0 | 0.00 ± 0.00 | 0.711 |
| AutoMix (V4, Bayes/POMDP) | 0.0 ± 0.0 | 0.00 ± 0.00 | 0.710 |
| Compute everything | 0.0 ± 0.0 | 0.00 ± 0.00 | 0.712 |

Always-SLM: cost 1.00, acc 0.504. Always-LLM: cost 5.34, acc 0.707 (split 0).

## Cost ratio LLM/SLM = 20

| Method | IBC lift (%) | Area lift (acc. points) | Max accuracy |
| --- | --- | --- | --- |
| Fixed V1 (answer confidence) | 48.3 ± 1.1 | 3.57 ± 0.10 | 0.711 |
| Ours: greedy VOI, per-query costs | 55.7 ± 3.4 | 3.54 ± 0.17 | 0.710 |
| Ours: greedy VOI | 50.7 ± 4.4 | 3.26 ± 0.36 | 0.710 |
| Fixed V4 (context-free verify) | 24.2 ± 0.8 | 1.84 ± 0.12 | 0.710 |
| AutoMix (V4, Bayes/POMDP) | 22.2 ± 1.8 | 1.61 ± 0.19 | 0.710 |
| Fixed V2 (self-consistency, best k) | 17.7 ± 1.4 | 1.49 ± 0.08 | 0.710 |
| Per-category map (oracle task) | 15.3 ± 1.8 | 1.41 ± 0.16 | 0.711 |
| Pre-generation router (V0) | 13.5 ± 4.5 | 0.95 ± 0.33 | 0.711 |
| Fixed V3 (retrieval verify) | 9.1 ± 1.1 | 0.81 ± 0.11 | 0.710 |
| Compute everything | 0.0 ± 0.0 | 0.00 ± 0.00 | 0.712 |

Always-SLM: cost 1.00, acc 0.504. Always-LLM: cost 21.34, acc 0.707 (split 0).

## Cost ratio LLM/SLM = 60

| Method | IBC lift (%) | Area lift (acc. points) | Max accuracy |
| --- | --- | --- | --- |
| Ours: greedy VOI, per-query costs | 62.9 ± 3.7 | 4.07 ± 0.26 | 0.710 |
| Ours: greedy VOI | 59.5 ± 5.1 | 3.90 ± 0.34 | 0.710 |
| Fixed V1 (answer confidence) | 51.6 ± 1.2 | 3.80 ± 0.10 | 0.711 |
| Per-category map (oracle task) | 31.1 ± 1.6 | 2.75 ± 0.14 | 0.711 |
| Fixed V2 (self-consistency, best k) | 31.2 ± 2.6 | 2.50 ± 0.16 | 0.710 |
| Fixed V4 (context-free verify) | 36.6 ± 1.2 | 2.48 ± 0.14 | 0.710 |
| AutoMix (V4, Bayes/POMDP) | 34.2 ± 1.9 | 2.22 ± 0.22 | 0.710 |
| Fixed V3 (retrieval verify) | 25.1 ± 1.7 | 2.12 ± 0.17 | 0.710 |
| Compute everything | 22.0 ± 3.5 | 2.00 ± 0.26 | 0.712 |
| Pre-generation router (V0) | 13.4 ± 4.5 | 0.94 ± 0.33 | 0.711 |

Always-SLM: cost 1.00, acc 0.504. Always-LLM: cost 64.03, acc 0.707 (split 0).
