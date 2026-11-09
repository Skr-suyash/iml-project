# Signal reliability

AUROC for predicting whether the SLM's answer is correct (0.5 = useless, 1 = perfect).
Cost in SLM-call units (unit cost model).

| Signal | Cost | Math | Factual | All |
| --- | --- | --- | --- | --- |
| V0 query-only prior (5-fold CV) | 0 | 0.558 | 0.703 | 0.672 |
| V1 answer confidence | 0 | 0.778 | 0.796 | 0.803 |
| V2 self-consistency, k=1 | 1 | 0.729 | 0.686 | 0.689 |
| V2 self-consistency, k=2 | 2 | 0.793 | 0.757 | 0.749 |
| V2 self-consistency, k=4 | 4 | 0.852 | 0.790 | 0.789 |
| V2 self-consistency, k=8 | 8 | 0.878 | 0.810 | 0.808 |
| V3 retrieval verify (4 votes) | 1.5 | 0.652 | 0.786 | 0.741 |
| V4 context-free verify (4 votes) | 1 | 0.705 | 0.715 | 0.740 |

## Latent type counts

| Task | easy | hard | unsolvable | slm_only |
| --- | --- | --- | --- | --- |
| math | 756 | 423 | 115 | 25 |
| factual | 485 | 290 | 578 | 89 |

## Mean tokens per call

| Task | slm_in | slm_out | sample_out | v3_in | v3_out | v4_in | v4_out | llm_in | llm_out |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| math | 165 | 157 | 144 | 587 | 20 | 293 | 20 | 165 | 161 |
| factual | 123 | 8 | 9 | 492 | 21 | 102 | 20 | 123 | 33 |

## Within-type Spearman correlation of V1, V2, V3, V4 (naive-Bayes check)

- easy: V1–V2 0.38, V1–V3 0.12, V1–V4 0.08, V2–V3 0.03, V2–V4 -0.20, V3–V4 0.12
- hard: V1–V2 0.26, V1–V3 0.28, V1–V4 0.42, V2–V3 -0.09, V2–V4 -0.20, V3–V4 0.48
- unsolvable: V1–V2 0.53, V1–V3 0.32, V1–V4 0.38, V2–V3 0.11, V2–V4 0.04, V3–V4 0.47
- slm_only: V1–V2 0.56, V1–V3 0.19, V1–V4 0.23, V2–V3 0.08, V2–V4 -0.14, V3–V4 0.28
