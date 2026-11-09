# Hidden-state probe as the free signal (token costs)

The probe (logistic regression on one layer of the SLM's hidden states plus V1) replaces V1. Pooling, layer and regularisation are chosen by 5-fold CV on each training split. Mean ± std over 5 splits.

| Variant | IBC lift, ratio 5 | IBC lift, ratio 20 | IBC lift, ratio 60 | Area lift, ratio 5 | Area lift, ratio 20 | Area lift, ratio 60 |
| --- | --- | --- | --- | --- | --- | --- |
| Ours (V1 log-prob), as in the main results | 34.7 ± 4.8 | 50.7 ± 4.4 | 59.5 ± 5.1 | 2.01 ± 0.37 | 3.26 ± 0.36 | 3.90 ± 0.34 |
| Ours + probe | 35.7 ± 4.3 | 54.2 ± 3.2 | 61.9 ± 4.0 | 2.22 ± 0.32 | 3.65 ± 0.19 | 4.17 ± 0.20 |
| Ours + probe, free signals only (no paid verifier) | 36.2 ± 4.0 | 50.2 ± 4.6 | 53.4 ± 4.8 | 2.28 ± 0.30 | 3.24 ± 0.32 | 3.46 ± 0.33 |
| Ours + probe + SLM-state prior (prefill charged) | 38.5 ± 5.3 | 57.6 ± 5.0 | 65.3 ± 5.6 | 2.25 ± 0.36 | 3.75 ± 0.28 | 4.29 ± 0.28 |
| Ours + probe, no gold labels | 28.0 ± 2.0 | 43.0 ± 2.9 | 47.6 ± 2.9 | 2.18 ± 0.19 | 3.32 ± 0.09 | 3.77 ± 0.16 |
| Ours + probe, 50 gold labels + calibrated agreement | 32.0 ± 5.4 | 49.0 ± 5.8 | 54.1 ± 6.5 | 2.05 ± 0.27 | 3.34 ± 0.30 | 3.85 ± 0.30 |
| Fixed V1 (log-prob threshold) | 32.9 ± 1.2 | 48.3 ± 1.1 | 51.6 ± 1.2 | 2.42 ± 0.09 | 3.57 ± 0.10 | 3.80 ± 0.10 |
| Fixed probe threshold | 31.9 ± 3.5 | 47.6 ± 3.8 | 50.8 ± 3.8 | 2.44 ± 0.25 | 3.64 ± 0.18 | 3.88 ± 0.17 |
| Pre-generation router (V0 sentence embedding) | 14.0 ± 4.5 | 13.5 ± 4.5 | 13.4 ± 4.5 | 0.99 ± 0.33 | 0.95 ± 0.33 | 0.94 ± 0.33 |
| Pre-generation router (SLM prompt state, prefill charged) | 15.8 ± 5.6 | 22.9 ± 6.5 | 24.5 ± 6.6 | 0.81 ± 0.30 | 1.28 ± 0.37 | 1.41 ± 0.39 |
| Ours, per-query costs | 38.5 ± 3.6 | 55.7 ± 3.4 | 62.9 ± 3.7 | 2.39 ± 0.12 | 3.54 ± 0.17 | 4.07 ± 0.26 |
| Ours + probe, per-query costs | 38.6 ± 3.4 | 57.1 ± 3.3 | 63.6 ± 4.2 | 2.57 ± 0.19 | 3.91 ± 0.15 | 4.39 ± 0.21 |
| Ours + probe + SLM-state prior, per-query costs (prefill charged) | 41.7 ± 3.7 | 62.5 ± 5.1 | 69.6 ± 6.3 | 2.55 ± 0.25 | 4.01 ± 0.24 | 4.49 ± 0.24 |

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
