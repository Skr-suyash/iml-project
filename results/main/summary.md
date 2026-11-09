# Results

## Cost ratio LLM/SLM = 3

| Method | IBC lift (%) | Area lift (acc. points) | Max accuracy |
| --- | --- | --- | --- |
| Ours: greedy VOI | 39.4 ± 4.5 | 2.37 ± 0.25 | 0.711 |
| Pre-generation router (V0) | 32.7 ± 3.4 | 2.32 ± 0.18 | 0.711 |
| Fixed V1 (answer confidence) | 2.4 ± 1.0 | 0.16 ± 0.08 | 0.711 |
| Fixed V2 (self-consistency, best k) | 0.0 ± 0.0 | 0.00 ± 0.00 | 0.710 |
| Fixed V3 (retrieval verify) | 0.0 ± 0.0 | 0.00 ± 0.00 | 0.710 |
| Fixed V4 (context-free verify) | 0.0 ± 0.0 | 0.00 ± 0.00 | 0.710 |
| Per-category map (oracle task) | 0.0 ± 0.0 | 0.00 ± 0.00 | 0.711 |
| AutoMix (V4, Bayes/POMDP) | 0.0 ± 0.0 | 0.00 ± 0.00 | 0.710 |
| Compute everything | 0.0 ± 0.0 | 0.00 ± 0.00 | 0.712 |

Always-SLM: cost 1.00, acc 0.504. Always-LLM: cost 3.00, acc 0.707 (split 0).

## Cost ratio LLM/SLM = 5

| Method | IBC lift (%) | Area lift (acc. points) | Max accuracy |
| --- | --- | --- | --- |
| Ours: greedy VOI | 48.3 ± 4.7 | 2.73 ± 0.28 | 0.711 |
| Pre-generation router (V0) | 32.7 ± 3.4 | 2.32 ± 0.18 | 0.711 |
| Fixed V1 (answer confidence) | 17.3 ± 0.9 | 1.39 ± 0.05 | 0.711 |
| Fixed V2 (self-consistency, best k) | 0.0 ± 0.0 | 0.00 ± 0.00 | 0.710 |
| Fixed V3 (retrieval verify) | 0.0 ± 0.0 | 0.00 ± 0.00 | 0.710 |
| Fixed V4 (context-free verify) | 0.0 ± 0.0 | 0.00 ± 0.00 | 0.710 |
| Per-category map (oracle task) | 0.0 ± 0.0 | 0.00 ± 0.00 | 0.711 |
| AutoMix (V4, Bayes/POMDP) | 0.0 ± 0.0 | 0.00 ± 0.00 | 0.710 |
| Compute everything | 0.0 ± 0.0 | 0.00 ± 0.00 | 0.712 |

Always-SLM: cost 1.00, acc 0.504. Always-LLM: cost 5.00, acc 0.707 (split 0).

## Cost ratio LLM/SLM = 10

| Method | IBC lift (%) | Area lift (acc. points) | Max accuracy |
| --- | --- | --- | --- |
| Ours: greedy VOI | 57.0 ± 3.7 | 3.32 ± 0.21 | 0.711 |
| Fixed V1 (answer confidence) | 28.5 ± 0.8 | 2.35 ± 0.06 | 0.711 |
| Pre-generation router (V0) | 32.7 ± 3.4 | 2.32 ± 0.18 | 0.711 |
| Fixed V2 (self-consistency, best k) | 11.2 ± 1.2 | 0.97 ± 0.08 | 0.710 |
| Per-category map (oracle task) | 6.0 ± 2.2 | 0.56 ± 0.20 | 0.711 |
| Fixed V3 (retrieval verify) | 0.0 ± 0.0 | 0.00 ± 0.00 | 0.710 |
| Fixed V4 (context-free verify) | 0.0 ± 0.0 | 0.00 ± 0.00 | 0.710 |
| AutoMix (V4, Bayes/POMDP) | 0.0 ± 0.0 | 0.00 ± 0.00 | 0.710 |
| Compute everything | 0.0 ± 0.0 | 0.00 ± 0.00 | 0.712 |

Always-SLM: cost 1.00, acc 0.504. Always-LLM: cost 10.00, acc 0.707 (split 0).

## Cost ratio LLM/SLM = 20

| Method | IBC lift (%) | Area lift (acc. points) | Max accuracy |
| --- | --- | --- | --- |
| Ours: greedy VOI | 63.4 ± 3.8 | 3.85 ± 0.26 | 0.710 |
| Fixed V1 (answer confidence) | 34.0 ± 0.8 | 2.79 ± 0.07 | 0.711 |
| Pre-generation router (V0) | 32.7 ± 3.4 | 2.32 ± 0.18 | 0.711 |
| Fixed V2 (self-consistency, best k) | 23.9 ± 1.5 | 1.96 ± 0.09 | 0.710 |
| Per-category map (oracle task) | 20.7 ± 2.0 | 1.87 ± 0.18 | 0.711 |
| Fixed V3 (retrieval verify) | 10.4 ± 1.1 | 0.92 ± 0.11 | 0.710 |
| Fixed V4 (context-free verify) | 7.1 ± 0.7 | 0.55 ± 0.07 | 0.710 |
| AutoMix (V4, Bayes/POMDP) | 6.1 ± 1.0 | 0.44 ± 0.09 | 0.710 |
| Compute everything | 0.0 ± 0.0 | -0.00 ± 0.00 | 0.712 |

Always-SLM: cost 1.00, acc 0.504. Always-LLM: cost 20.00, acc 0.707 (split 0).

## Cost ratio LLM/SLM = 60

| Method | IBC lift (%) | Area lift (acc. points) | Max accuracy |
| --- | --- | --- | --- |
| Ours: greedy VOI | 72.0 ± 4.6 | 4.47 ± 0.26 | 0.710 |
| Per-category map (oracle task) | 39.1 ± 5.7 | 3.14 ± 0.20 | 0.711 |
| Fixed V1 (answer confidence) | 37.6 ± 0.8 | 3.06 ± 0.07 | 0.711 |
| Fixed V2 (self-consistency, best k) | 39.4 ± 2.8 | 3.05 ± 0.15 | 0.710 |
| Compute everything | 27.6 ± 3.2 | 2.40 ± 0.20 | 0.712 |
| Pre-generation router (V0) | 32.7 ± 3.4 | 2.32 ± 0.18 | 0.711 |
| Fixed V3 (retrieval verify) | 21.6 ± 1.7 | 1.88 ± 0.16 | 0.710 |
| Fixed V4 (context-free verify) | 18.4 ± 0.9 | 1.30 ± 0.09 | 0.710 |
| AutoMix (V4, Bayes/POMDP) | 17.2 ± 1.4 | 1.15 ± 0.12 | 0.710 |

Always-SLM: cost 1.00, acc 0.504. Always-LLM: cost 60.00, acc 0.707 (split 0).

## Cost ratio LLM/SLM = 200

| Method | IBC lift (%) | Area lift (acc. points) | Max accuracy |
| --- | --- | --- | --- |
| Ours: greedy VOI | 80.1 ± 5.7 | 4.88 ± 0.26 | 0.711 |
| Compute everything | 61.8 ± 5.1 | 4.47 ± 0.23 | 0.712 |
| Per-category map (oracle task) | 59.7 ± 4.2 | 4.02 ± 0.14 | 0.711 |
| Fixed V2 (self-consistency, best k) | 49.9 ± 2.9 | 3.78 ± 0.11 | 0.710 |
| Fixed V1 (answer confidence) | 38.8 ± 0.8 | 3.15 ± 0.08 | 0.711 |
| Pre-generation router (V0) | 32.7 ± 3.4 | 2.32 ± 0.18 | 0.711 |
| Fixed V3 (retrieval verify) | 25.8 ± 1.7 | 2.21 ± 0.17 | 0.710 |
| Fixed V4 (context-free verify) | 22.9 ± 0.9 | 1.55 ± 0.10 | 0.710 |
| AutoMix (V4, Bayes/POMDP) | 21.5 ± 1.5 | 1.39 ± 0.13 | 0.710 |

Always-SLM: cost 1.00, acc 0.504. Always-LLM: cost 200.00, acc 0.707 (split 0).
