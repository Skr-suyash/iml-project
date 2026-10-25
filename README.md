# Choosing What to Verify in Context-Free LLM Cascades

Semester project extending [AutoMix](https://arxiv.org/abs/2310.12963). A small model (SLM)
answers first; the router decides, per query and under a cost budget, **which verification
signal to buy** (if any) before accepting the answer or escalating to a large model (LLM).

| Signal | What it is | Cost (SLM-call units) |
| --- | --- | --- |
| V0 | Query-only prior: embedding + logistic regression | ~0 |
| V1 | SLM answer confidence (mean token log-prob) | 0, free with generation |
| V2 | Self-consistency: agreement of extra samples, drawn one at a time | 1 per sample |
| V3 | Retrieve Wikipedia passages, then AutoMix-style verification against them | ~1.5 |
| V4 | AutoMix-style verification with no context (the AutoMix baseline) | 1 |

## Results in one paragraph

On 2,761 GSM8K + TriviaQA questions (Qwen2.5-1.5B → Qwen2.5-7B), the greedy value-of-information
router beats AutoMix, every fixed verifier, a per-category verifier map given the true task, and a
pre-generation router at every cost ratio from 3 to 200 (IBC lift 39–80% vs AutoMix 0–22%;
paired bootstrap p < 0.001 except a tie with the pre-generation router at ratio 3). Most of
the gain comes from combining the free signals (query prior + answer confidence); self-consistency
is the only paid verifier that pays; retrieval and context-free verification add nothing beyond
it.

Follow-ups (report §4.6):
- **Hidden-state probe.** A linear probe on the SLM's hidden states replaces token log-prob as
  the free signal: AUROC 0.82–0.84 vs 0.78–0.80, +0.4–0.5 area-lift points, and a win over the
  confidence threshold under token costs at ratios 20 and 60.
- **Calibrated LLM-agreement labels.** These cut the gold labels needed from 200+ to about 50.

Full write-up: [`report/report.md`](report/report.md).

## Layout

```
src/vcascade/
  data.py prompts.py grading.py backends.py retrieval.py generate.py   # GPU side: build the cache
  signals.py costs.py belief.py routers.py evaluate.py experiment.py    # CPU side: routers + evaluation
  simulate.py                                                          # synthetic cache for testing
  ablations.py                                                         # signal AUROC, exact vs greedy, bootstrap, ablations, ...
  probe.py                                                             # hidden-state probe signal, SLM-state prior
scripts/build_notebook.py   # builds notebooks/generate.ipynb (self-contained, for Kaggle / Colab)
scripts/kaggle_run.py       # push the notebook to Kaggle, upload inputs datasets, fetch outputs
scripts/make_figures.py     # report figures
scripts/paired_splits.py    # split-paired comparison of two methods in a results json
results/                    # main, token_cost, robust_smollm2, ablations (summary.md / *.md in each)
report/report.md            # the write-up
tests/                      # pytest
```

## 1. Generate the cache (GPU: Kaggle or Colab)

Done for this project with `scripts/kaggle_run.py` in two phases (see its docstring): GPU stages
on Kaggle, Wikipedia retrieval locally (set `WIKI_CONTACT` to an email or URL; Wikimedia allows
200 requests/min with it, 10 without), then verification on Kaggle.


Upload `notebooks/generate.ipynb`, pick a **T4** GPU (vLLM does not run on the P100), turn
internet on, run with `PILOT = True` first, then `PILOT = False`. Download `cache.zip` and
unzip it into `cache/`. After editing anything in `src/vcascade/`, rebuild the notebook with
`python scripts/build_notebook.py`.

Default models: SLM `Qwen/Qwen2.5-1.5B-Instruct`, LLM `Qwen/Qwen2.5-7B-Instruct-AWQ`.
A hosted 70B-class LLM can replace the 7B through `--backend openai --base-url ...`
(API key in `LLM_API_KEY`).

## 2. Run the experiments (CPU, local)

```
python -m venv .venv && .venv/Scripts/pip install -e ".[dev]"
python -m vcascade.experiment --cache cache --out results            # real data
python -m vcascade.experiment --synthetic --out results/synthetic    # pipeline check
python -m pytest
```

Outputs: `results/summary.md` (IBC lift and area lift per method and cost ratio, mean ± std
over 5 splits), `curves_r*.png`, `lift_vs_ratio.png`, `results.json`.

## Methods compared

- Always-SLM, always-LLM, random mixing (reference line)
- Fixed verifier + threshold: V1, V2 (k = 8), V3, V4
- Per-category map given the true task label (math → V2, factual → V3)
- Pre-generation router (V0 only)
- AutoMix: one V4 observation, Bayes decision (for two models AutoMix's POMDP reduces to this)
- Compute everything: all signals for every query, then decide
- **Ours:** greedy value-of-information router; exact expectimax planner (`--exact`, on a subsample)
