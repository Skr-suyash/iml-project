# Hidden-state probe as the free signal (unit costs)

The probe (logistic regression on one layer of the SLM's hidden states plus V1) replaces V1. Pooling, layer and regularisation are chosen by 5-fold CV on each training split. Mean ± std over 5 splits.

| Variant | IBC lift, ratio 5 | IBC lift, ratio 20 | IBC lift, ratio 60 | Area lift, ratio 5 | Area lift, ratio 20 | Area lift, ratio 60 |
| --- | --- | --- | --- | --- | --- | --- |
| Ours (V1 log-prob), as in the main results | 48.3 ± 4.7 | 63.4 ± 3.8 | 72.0 ± 4.6 | 2.73 ± 0.28 | 3.85 ± 0.26 | 4.47 ± 0.26 |
| Ours + probe | 53.6 ± 4.7 | 70.6 ± 3.2 | 79.0 ± 4.4 | 3.09 ± 0.25 | 4.38 ± 0.15 | 4.87 ± 0.17 |
| Ours + probe, free signals only (no paid verifier) | 54.0 ± 4.6 | 69.8 ± 5.3 | 73.1 ± 5.3 | 3.13 ± 0.23 | 4.01 ± 0.25 | 4.22 ± 0.26 |
| Ours + probe + SLM-state prior | 59.4 ± 3.2 | 76.2 ± 3.8 | 83.6 ± 4.8 | 3.44 ± 0.17 | 4.55 ± 0.17 | 5.02 ± 0.18 |
| Ours + probe, no gold labels | 17.2 ± 2.7 | 35.8 ± 2.0 | 41.7 ± 3.2 | 1.42 ± 0.27 | 2.88 ± 0.14 | 3.43 ± 0.20 |
| Ours + probe, 50 gold labels + calibrated agreement | 41.0 ± 8.6 | 61.0 ± 8.7 | 67.9 ± 9.7 | 2.41 ± 0.34 | 3.76 ± 0.30 | 4.31 ± 0.33 |
| Fixed V1 (log-prob threshold) | 17.3 ± 0.9 | 34.0 ± 0.8 | 37.6 ± 0.8 | 1.39 ± 0.05 | 2.79 ± 0.07 | 3.06 ± 0.07 |
| Fixed probe threshold | 22.2 ± 1.7 | 39.1 ± 1.9 | 42.6 ± 1.9 | 1.70 ± 0.17 | 3.08 ± 0.12 | 3.35 ± 0.11 |
| Pre-generation router (V0 sentence embedding) | 32.7 ± 3.4 | 32.7 ± 3.4 | 32.7 ± 3.4 | 2.32 ± 0.18 | 2.32 ± 0.18 | 2.32 ± 0.18 |
| Pre-generation router (SLM prompt state) | 45.4 ± 3.9 | 45.4 ± 3.9 | 45.4 ± 3.9 | 2.80 ± 0.21 | 2.80 ± 0.21 | 2.80 ± 0.21 |

## Signal AUROC for SLM correctness (5-fold CV over all queries, C = 0.01)

| Signal | Math | Factual |
| --- | --- | --- |
| V1 answer confidence | 0.778 | 0.796 |
| V2 self-consistency, k=8 | 0.878 | 0.810 |
| Probe, last token + V1, layer 1/4 | 0.767 | 0.686 |
| Probe, last token + V1, layer 2/4 | 0.816 | 0.816 |
| Probe, last token + V1, layer 3/4 | 0.832 | 0.817 |
| Probe, last token + V1, layer 4/4 | 0.802 | 0.809 |
| Probe, mean tokens + V1, layer 1/4 | 0.825 | 0.746 |
| Probe, mean tokens + V1, layer 2/4 | 0.853 | 0.844 |
| Probe, mean tokens + V1, layer 3/4 | 0.867 | 0.823 |
| Probe, mean tokens + V1, layer 4/4 | 0.855 | 0.779 |
| Probe on prompt state (query only), layer 1/4 | 0.612 | 0.699 |
| Probe on prompt state (query only), layer 2/4 | 0.649 | 0.738 |
| Probe on prompt state (query only), layer 3/4 | 0.712 | 0.759 |
| Probe on prompt state (query only), layer 4/4 | 0.675 | 0.733 |

Probe chosen on each training split, AUROC on its test split (mean over splits): math 0.844, factual 0.822. Trained on LLM agreement instead of gold labels: math 0.849, factual 0.824.

Chosen configurations: split 0: mean, layer 2/4, C=0.01; split 1: mean, layer 2/4, C=0.01; split 2: mean, layer 3/4, C=0.001; split 3: mean, layer 3/4, C=0.01; split 4: mean, layer 2/4, C=0.01
