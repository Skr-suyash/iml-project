"""Follow-up analyses on the cached signals. Each subcommand writes <name>.json and <name>.md.

    python -m vcascade.ablations signals    --cache cache --out results/ablations
    python -m vcascade.ablations exact      ...   # H5: exact planner vs greedy, on a subsample
    python -m vcascade.ablations bootstrap  ...   # paired bootstrap, ours vs every baseline
    python -m vcascade.ablations ablate     ...   # remove one component of the router at a time
    python -m vcascade.ablations trainsize  ...   # 50 .. all training examples
    python -m vcascade.ablations shift      ...   # train on one task mix, test on another
    python -m vcascade.ablations labelfree  ...   # train with the LLM's answers instead of gold labels
    python -m vcascade.ablations probe      ...   # hidden-state probe as the free signal (needs hidden.npz)

`labelfree` and `probe` take --cost-model unit|token.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import TYPES, evaluate, routers
from .belief import BeliefModel
from .costs import CostModel
from .experiment import METHOD_LABELS, compute_outcomes, split
from .signals import load

BASELINES = ["fixed_v1", "fixed_v2", "fixed_v3", "fixed_v4", "category_map", "pregen", "automix", "compute_all"]


def _args(**kw):
    base = dict(groups=4, seed=0, n_lambda=24, exact=False, exact_n=400)
    base.update(kw)
    return SimpleNamespace(**base)


def _summ(outs, te, cm):
    slm, llm = routers.always_slm(te, cm).point, routers.always_llm(te, cm).point
    s = evaluate.summarize(outs, slm, llm)
    return {"ibc_lift": s["ibc_lift"], "area_lift": s["area_lift"]}


def _fmt(vals):
    return f"{np.mean(vals):.1f} ± {np.std(vals):.1f}"


def _fmt2(vals):
    return f"{np.mean(vals):.2f} ± {np.std(vals):.2f}"


def _save(out: Path, name: str, data, md: str):
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}.json").write_text(json.dumps(data, indent=1), encoding="utf-8")
    (out / f"{name}.md").write_text(md, encoding="utf-8")
    print(md)


# ------------------------------------------------------------------ signals

def cmd_signals(sig, out, a):
    """How well does each signal predict SLM correctness, per task, and what does it cost?"""
    v0 = cross_val_predict(LogisticRegression(C=1.0, max_iter=2000), sig.emb, sig.slm_correct,
                           cv=5, method="predict_proba")[:, 1]
    scores = {"V0 query-only prior (5-fold CV)": v0, "V1 answer confidence": sig.v1}
    for k in (1, 2, 4, 8):
        scores[f"V2 self-consistency, k={k}"] = sig.v2[:, :k].mean(1)
    scores["V3 retrieval verify (4 votes)"] = sig.v3.astype(float)
    scores["V4 context-free verify (4 votes)"] = sig.v4.astype(float)
    cost = {"V0": 0.0, "V1": 0.0, "V3": 1.5, "V4": 1.0}

    rows, data = [], {}
    for name, sc in scores.items():
        r = {}
        for task in ("math", "factual", "all"):
            m = np.ones(len(sig), bool) if task == "all" else sig.task == task
            r[task] = float(roc_auc_score(sig.slm_correct[m], sc[m]))
        data[name] = r
        c = cost.get(name[:2], None)
        if name.startswith("V2"):
            c = float(name.split("k=")[1])
        rows.append(f"| {name} | {c:g} | {r['math']:.3f} | {r['factual']:.3f} | {r['all']:.3f} |")

    tok = {t: {k: float(np.mean(v[sig.task == t])) for k, v in sig.tokens.items()} for t in ("math", "factual")}
    z = sig.z
    type_counts = {t: {TYPES[i]: int(((z == i) & (sig.task == t)).sum()) for i in range(4)}
                   for t in ("math", "factual")}

    # naive-Bayes check: are signals independent given the type?
    from scipy.stats import spearmanr
    feats = np.column_stack([sig.v1, sig.v2.mean(1), sig.v3, sig.v4])
    corr = {}
    for i, t in enumerate(TYPES):
        m = z == i
        if m.sum() > 20:
            rho = spearmanr(feats[m]).correlation
            corr[t] = np.round(rho, 2).tolist()

    md = ["# Signal reliability", "",
          "AUROC for predicting whether the SLM's answer is correct (0.5 = useless, 1 = perfect).",
          "Cost in SLM-call units (unit cost model).", "",
          "| Signal | Cost | Math | Factual | All |", "| --- | --- | --- | --- | --- |", *rows, "",
          "## Latent type counts", "", "| Task | " + " | ".join(TYPES) + " |", "| --- |" + " --- |" * 4]
    for t, c in type_counts.items():
        md.append(f"| {t} | " + " | ".join(str(c[x]) for x in TYPES) + " |")
    md += ["", "## Mean tokens per call", "", "| Task | " + " | ".join(tok['math']) + " |",
           "| --- |" + " --- |" * len(tok['math'])]
    for t, d in tok.items():
        md.append(f"| {t} | " + " | ".join(f"{v:.0f}" for v in d.values()) + " |")
    md += ["", "## Within-type Spearman correlation of V1, V2, V3, V4 (naive-Bayes check)", ""]
    for t, rho in corr.items():
        md.append(f"- {t}: V1–V2 {rho[0][1]:.2f}, V1–V3 {rho[0][2]:.2f}, V1–V4 {rho[0][3]:.2f}, "
                  f"V2–V3 {rho[1][2]:.2f}, V2–V4 {rho[1][3]:.2f}, V3–V4 {rho[2][3]:.2f}")
    _save(out, "signals", {"auroc": data, "tokens": tok, "types": type_counts, "corr": corr}, "\n".join(md) + "\n")


# ------------------------------------------------------------------ exact vs greedy

def cmd_exact(sig, out, a):
    res = {}
    for r in a.ratios:
        res[r] = {"exact (max 4 samples)": [], "greedy (max 4 samples)": [], "greedy (max 8 samples)": []}
        for sp in range(a.splits):
            t0 = time.time()
            tr_i, te_i = split(len(sig), 0.3, sp)
            tr, te = sig.subset(tr_i), sig.subset(te_i)
            sub = np.random.default_rng(sp).permutation(len(te))[:a.n]
            te_s = te.subset(sub)
            cm = CostModel.unit(len(te_s), r)
            model = BeliefModel(4, seed=0).fit(tr)
            lams = routers.lambda_grid(cm, 12)
            te4 = te_s.subset(np.arange(len(te_s)))
            te4.v2 = te4.v2[:, :4]
            res[r]["exact (max 4 samples)"].append(_summ([routers.exact_planner(model, te_s, cm, l, max_k=4) for l in lams], te_s, cm))
            res[r]["greedy (max 4 samples)"].append(_summ([routers.greedy_voi(model, te4, cm, l) for l in lams], te_s, cm))
            res[r]["greedy (max 8 samples)"].append(_summ([routers.greedy_voi(model, te_s, cm, l) for l in lams], te_s, cm))
            print(f"exact: ratio {r} split {sp} {time.time() - t0:.0f}s", flush=True)
    md = ["# Exact planner vs. greedy (H5)", "",
          f"{a.n}-query test subsample per split, {a.splits} splits, 12 lambda values.", "",
          "| Cost ratio | Method | IBC lift (%) | Area lift (acc. points) |", "| --- | --- | --- | --- |"]
    for r, d in res.items():
        for m, v in d.items():
            md.append(f"| {r} | {m} | {_fmt([x['ibc_lift'] for x in v])} | {_fmt2([x['area_lift'] for x in v])} |")
    _save(out, "exact", res, "\n".join(md) + "\n")


# ------------------------------------------------------------------ bootstrap

def _thin(outs, max_points=150):
    if len(outs) <= max_points:
        return outs
    outs = sorted(outs, key=lambda o: o.point[0])
    keep = np.unique(np.linspace(0, len(outs) - 1, max_points).round().astype(int))
    return [outs[i] for i in keep]


def cmd_bootstrap(sig, out, a):
    rng = np.random.default_rng(0)
    res = {}
    for r in a.ratios:
        diffs = {m: {"area": [], "ibc": []} for m in BASELINES}
        for sp in range(a.splits):
            t0 = time.time()
            tr_i, te_i = split(len(sig), 0.3, sp)
            tr, te = sig.subset(tr_i), sig.subset(te_i)
            cm = CostModel.unit(len(te), r)
            outs, _, _ = compute_outcomes(tr, te, cm, _args())
            n = len(te)
            mats = {}
            for m in BASELINES + ["greedy_voi"]:
                o = _thin(outs[m])
                mats[m] = (np.stack([x.cost for x in o]), np.stack([x.perf for x in o]))
            ref_c = np.stack([cm.per_query["gen"], cm.per_query["llm"]])
            ref_p = np.stack([te.slm_correct, te.llm_correct]).astype(float)
            W = rng.multinomial(n, np.full(n, 1 / n), size=a.boot).T / n  # (n, B) resampling weights
            rc, rp = ref_c @ W, ref_p @ W                                  # (2, B)
            metric = {}
            for m, (C, P) in mats.items():
                cc, pp = C @ W, P @ W                                      # (points, B)
                area, ibc = np.zeros(a.boot), np.zeros(a.boot)
                for b in range(a.boot):
                    slm, llm = (rc[0, b], rp[0, b]), (rc[1, b], rp[1, b])
                    fr = evaluate.frontier(list(zip(cc[:, b], pp[:, b])) + [slm, llm])
                    area[b] = evaluate.area_lift(fr, slm, llm)
                    ibc[b] = evaluate.ibc_lift(fr, slm, llm)
                metric[m] = (area, ibc)
            for m in BASELINES:
                diffs[m]["area"].append(metric["greedy_voi"][0] - metric[m][0])
                diffs[m]["ibc"].append(metric["greedy_voi"][1] - metric[m][1])
            print(f"bootstrap: ratio {r} split {sp} {time.time() - t0:.0f}s", flush=True)
        res[r] = {}
        for m in BASELINES:
            row = {}
            for k in ("area", "ibc"):
                d = np.mean(diffs[m][k], axis=0)  # average over splits, per bootstrap draw
                lo, hi = np.percentile(d, [2.5, 97.5])
                row[k] = {"mean": float(d.mean()), "lo": float(lo), "hi": float(hi), "p_le_0": float((d <= 0).mean())}
            res[r][m] = row
    md = ["# Paired bootstrap: ours (greedy VOI) minus each baseline", "",
          f"{a.boot} resamples of the test queries per split, averaged over {a.splits} splits. "
          "Mean difference with 95% interval; p = share of resamples where ours is not better.", "",
          "| Cost ratio | Baseline | Area lift diff (points) | p | IBC lift diff (%) | p |",
          "| --- | --- | --- | --- | --- | --- |"]
    for r, d in res.items():
        for m, row in d.items():
            ar, ib = row["area"], row["ibc"]
            md.append(f"| {r} | {METHOD_LABELS[m]} | {ar['mean']:.2f} [{ar['lo']:.2f}, {ar['hi']:.2f}] | "
                      f"{ar['p_le_0']:.3f} | {ib['mean']:.1f} [{ib['lo']:.1f}, {ib['hi']:.1f}] | {ib['p_le_0']:.3f} |")
    _save(out, "bootstrap", res, "\n".join(md) + "\n")


# ------------------------------------------------------------------ ablations

VARIANTS = {
    "Full router": {},
    "− V1 (answer confidence)": {"use_v1": False},
    "− V2 (self-consistency)": {"menu": ("v3", "v4")},
    "− V3 (retrieval verify)": {"menu": ("v2", "v4")},
    "− V4 (context-free verify)": {"menu": ("v2", "v3")},
    "− V0 prior (base rates instead)": {"use_prior": False},
    "− skip-the-SLM action": {"allow_skip": False},
    "− batch lookahead for V2 (1 sample)": {"v2_batches": (1,)},
    "− query groups (1 group)": {"groups": 1},
    "V1 modelled with a KDE instead of bins": {"v1_model": "kde"},
    "Free signals only (V0 + V1, no paid verifier)": {"menu": ()},
    "Only paid verifier: V2": {"menu": ("v2",)},
}


def cmd_ablate(sig, out, a):
    res = {r: {v: [] for v in list(VARIANTS) + ["Compute everything (naive Bayes)",
                                                 "Compute everything (learned combiner)"]} for r in a.ratios}
    for sp in range(a.splits):
        t0 = time.time()
        tr_i, te_i = split(len(sig), 0.3, sp)
        tr, te = sig.subset(tr_i), sig.subset(te_i)
        models = {(g, v): BeliefModel(g, seed=0, v1_model=v).fit(tr) for g, v in ((1, "bins"), (4, "bins"), (4, "kde"))}
        for r in a.ratios:
            cm = CostModel.unit(len(te), r)
            lams = routers.lambda_grid(cm, 24)
            for name, kw in VARIANTS.items():
                kw = dict(kw)
                model = models[(kw.pop("groups", 4), kw.pop("v1_model", "bins"))]
                res[r][name].append(_summ([routers.greedy_voi(model, te, cm, l, **kw) for l in lams], te, cm))
            res[r]["Compute everything (naive Bayes)"].append(
                _summ(routers.posterior_gain_sweep(models[(4, "bins")], te, cm, ("v1", "v2", "v3", "v4")), te, cm))
            res[r]["Compute everything (learned combiner)"].append(
                _summ(routers.learned_combiner(tr, te, cm), te, cm))
        print(f"ablate: split {sp} {time.time() - t0:.0f}s", flush=True)
    md = ["# Ablations of the greedy VOI router", "", f"Mean ± std over {a.splits} splits.", "",
          "| Variant | " + " | ".join(f"IBC lift, ratio {r}" for r in a.ratios) + " | "
          + " | ".join(f"Area lift, ratio {r}" for r in a.ratios) + " |",
          "| --- |" + " --- |" * (2 * len(a.ratios))]
    for v in res[a.ratios[0]]:
        md.append(f"| {v} | " + " | ".join(_fmt([x["ibc_lift"] for x in res[r][v]]) for r in a.ratios) + " | "
                  + " | ".join(_fmt2([x["area_lift"] for x in res[r][v]]) for r in a.ratios) + " |")
    _save(out, "ablate", res, "\n".join(md) + "\n")


# ------------------------------------------------------------------ training size and shift

def _methods_on(tr, te, cm):
    outs, _, _ = compute_outcomes(tr, te, cm, _args())
    keep = ["greedy_voi", "fixed_v1", "fixed_v2", "category_map", "pregen", "automix", "compute_all"]
    return {m: _summ(outs[m], te, cm) for m in keep}


def cmd_trainsize(sig, out, a):
    sizes = [50, 100, 200, 400, "all"]
    res = {r: {str(n): {} for n in sizes} for r in a.ratios}
    for sp in range(a.splits):
        t0 = time.time()
        tr_i, te_i = split(len(sig), 0.3, sp)
        te = sig.subset(te_i)
        perm = np.random.default_rng(100 + sp).permutation(tr_i)
        for n in sizes:
            tr = sig.subset(np.sort(perm if n == "all" else perm[:n]))
            for r in a.ratios:
                for m, v in _methods_on(tr, te, CostModel.unit(len(te), r)).items():
                    res[r][str(n)].setdefault(m, []).append(v)
        print(f"trainsize: split {sp} {time.time() - t0:.0f}s", flush=True)
    md = ["# Training-set size", "", f"Fixed 70% test split; train on a random subset of the 30% train split. "
          f"Area lift (accuracy points), mean ± std over {a.splits} splits.", ""]
    for r in a.ratios:
        ms = list(res[r]["all"])
        md += [f"## Cost ratio {r}", "", "| n train | " + " | ".join(METHOD_LABELS[m] for m in ms) + " |",
               "| --- |" + " --- |" * len(ms)]
        for n in sizes:
            md.append(f"| {n} | " + " | ".join(_fmt2([x["area_lift"] for x in res[r][str(n)][m]]) for m in ms) + " |")
        md.append("")
    _save(out, "trainsize", res, "\n".join(md) + "\n")


def cmd_shift(sig, out, a):
    math_i, fact_i = np.where(sig.task == "math")[0], np.where(sig.task == "factual")[0]
    n_tr = 830
    res = {}
    for direction, (tr_pool, te_pool) in {"train 80% math → test 80% factual": (math_i, fact_i),
                                          "train 80% factual → test 80% math": (fact_i, math_i)}.items():
        res[direction] = {r: {} for r in a.ratios}
        for sp in range(a.splits):
            rng = np.random.default_rng(200 + sp)
            maj, mino = rng.permutation(tr_pool), rng.permutation(te_pool)
            tr_idx = np.concatenate([maj[:int(0.8 * n_tr)], mino[:n_tr - int(0.8 * n_tr)]])
            rest_maj, rest_min = maj[int(0.8 * n_tr):], mino[n_tr - int(0.8 * n_tr):]
            n_te_min = len(rest_min)  # all remaining of the test-majority task
            te_idx = np.concatenate([rest_min, rest_maj[:int(n_te_min / 4)]])  # 80 / 20
            tr, te = sig.subset(np.sort(tr_idx)), sig.subset(np.sort(te_idx))
            for r in a.ratios:
                for m, v in _methods_on(tr, te, CostModel.unit(len(te), r)).items():
                    res[direction][r].setdefault(m, []).append(v)
            print(f"shift: {direction} split {sp}", flush=True)
    md = ["# Task-mix shift", "", "Routers are trained on one task mix and tested on the opposite mix. "
          f"Area lift (accuracy points), mean ± std over {a.splits} seeds.", ""]
    for d, byr in res.items():
        ms = list(byr[a.ratios[0]])
        md += [f"## {d}", "", "| Cost ratio | " + " | ".join(METHOD_LABELS[m] for m in ms) + " |",
               "| --- |" + " --- |" * len(ms)]
        for r in a.ratios:
            md.append(f"| {r} | " + " | ".join(_fmt2([x["area_lift"] for x in byr[r][m]]) for m in ms) + " |")
        md.append("")
    _save(out, "shift", res, "\n".join(md) + "\n")


# ------------------------------------------------------------------ label-free training and probe

def _cost(sig, idx, r, kind):
    if kind == "unit":
        return CostModel.unit(len(idx), r)
    return CostModel.token(sig.tokens, llm_price=(r, r)).subset(idx)


def _table(title, intro, res, ratios):
    md = [f"# {title}", "", intro, "",
          "| Variant | " + " | ".join(f"IBC lift, ratio {r}" for r in ratios) + " | "
          + " | ".join(f"Area lift, ratio {r}" for r in ratios) + " |",
          "| --- |" + " --- |" * (2 * len(ratios))]
    for v in res[ratios[0]]:
        md.append(f"| {v} | " + " | ".join(_fmt([x["ibc_lift"] for x in res[r][v]]) for r in ratios) + " | "
                  + " | ".join(_fmt2([x["area_lift"] for x in res[r][v]]) for r in ratios) + " |")
    return md


def cmd_labelfree(sig, out, a):
    """Train the router without gold answers: the LLM's answer stands in for the truth."""
    from .signals import llm_agreement, pseudo_labeled, soft_labeled

    agree = llm_agreement(sig)
    quality = {}
    for t in ("math", "factual", "all"):
        m = np.ones(len(sig), bool) if t == "all" else sig.task == t
        quality[t] = {"label_accuracy": float((agree[m] == sig.slm_correct[m]).mean()),
                      "agree_but_slm_wrong": float(((agree == 1) & (sig.slm_correct == 0))[m].mean()),
                      "disagree_but_slm_right": float(((agree == 0) & (sig.slm_correct == 1))[m].mean())}
    res = {r: {} for r in a.ratios}
    for sp in range(a.splits):
        t0 = time.time()
        tr_i, te_i = split(len(sig), 0.3, sp)
        tr, te = sig.subset(tr_i), sig.subset(te_i)
        perm = np.random.default_rng(100 + sp).permutation(tr_i)  # same subsets as `trainsize`
        pseudo = pseudo_labeled(tr)
        models = {
            "Ours, gold labels (all 828)": BeliefModel(4, seed=0).fit(tr),
            "Ours, gold labels (200)": BeliefModel(4, seed=0).fit(sig.subset(np.sort(perm[:200]))),
            "Ours, gold labels (100)": BeliefModel(4, seed=0).fit(sig.subset(np.sort(perm[:100]))),
            "Ours, gold labels (50)": BeliefModel(4, seed=0).fit(sig.subset(np.sort(perm[:50]))),
            "Ours, no gold labels (LLM answer as truth)": BeliefModel(4, seed=0).fit(pseudo),
            "Ours, 50 gold labels + LLM answer for the rest (hard labels)":
                BeliefModel(4, seed=0).fit(pseudo_labeled(tr, keep_true=np.isin(tr_i, perm[:50]))),
        }
        for n in (20, 50, 100):  # calibrated stand-in labels: P(type | agreement, group) from n gold labels
            soft, w = soft_labeled(tr, np.isin(tr_i, perm[:n]), seed=0)
            models[f"Ours, {n} gold labels + calibrated LLM agreement"] = BeliefModel(4, seed=0).fit(soft, w)
        for r in a.ratios:
            cm = _cost(sig, te_i, r, a.cost_model)
            lams = routers.lambda_grid(cm, 24)
            row = {name: [routers.greedy_voi(m, te, cm, l) for l in lams] for name, m in models.items()}
            row["Pre-generation router, no gold labels"] = routers.pregen_router(
                models["Ours, no gold labels (LLM answer as truth)"], te, cm)
            row["AutoMix, no gold labels"] = routers.posterior_gain_sweep(
                BeliefModel(1, seed=0).fit(pseudo), te, cm, ("v4",), use_prior=False)
            row["AutoMix, gold labels"] = routers.posterior_gain_sweep(
                BeliefModel(1, seed=0).fit(tr), te, cm, ("v4",), use_prior=False)
            row["Fixed V1 (needs no labels)"] = routers.fixed_verifier("v1", te, cm)
            row["Fixed V2, best k (needs no labels)"] = routers.fixed_verifier("v2", te, cm)
            for name, o in row.items():
                res[r].setdefault(name, []).append(_summ(o, te, cm))
        print(f"labelfree: split {sp} {time.time() - t0:.0f}s", flush=True)
    md = _table(f"Training without gold labels ({a.cost_model} costs)",
                "The router is trained with the LLM's answer as the truth: SLM correct := its answer "
                "matches the LLM's. Needs both models run once on the 828 training queries, no gold answers. "
                f"Evaluated against gold answers. Mean ± std over {a.splits} splits.", res, a.ratios)
    md += ["", "## Stand-in label quality (all 2,761 queries)", "",
           "| Task | Stand-in = gold SLM correctness | Agree but SLM wrong | Disagree but SLM right |",
           "| --- | --- | --- | --- |"]
    for t, q in quality.items():
        md.append(f"| {t} | {q['label_accuracy']:.3f} | {q['agree_but_slm_wrong']:.3f} | "
                  f"{q['disagree_but_slm_right']:.3f} |")
    _save(out, f"labelfree_{a.cost_model}", {"quality": quality, "results": res}, "\n".join(md) + "\n")


def _cv_auroc(x, y, seed=0, C=1e-2):
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000))
    oof = cross_val_predict(clf, x, y, cv=StratifiedKFold(5, shuffle=True, random_state=seed),
                            method="predict_proba")[:, 1]
    return oof


def cmd_probe(sig, out, a):
    """Replace V1 with a probe on the SLM's hidden states; optionally the V0 prior too."""
    from dataclasses import replace

    from .probe import POOLS, features, slm_prior, with_probe
    from .signals import pseudo_labeled, soft_labeled

    if sig.hidden is None:
        raise SystemExit("no hidden.npz in the cache; run the `hidden` generation stage first")
    tasks = ("math", "factual")

    def by_task(y, score, task):
        return {t: float(roc_auc_score(y[task == t], score[task == t])) for t in tasks}

    # Progress is checkpointed after the AUROC table and after every split; a rerun resumes.
    partial = out / f"probe_{a.cost_model}.partial.json"
    state = json.loads(partial.read_text(encoding="utf-8")) if partial.exists() else {}

    def checkpoint():
        out.mkdir(parents=True, exist_ok=True)
        partial.write_text(json.dumps({"auroc": auroc, "configs": configs,
                                       "res": {str(r): v for r, v in res.items()}}), encoding="utf-8")

    # 1) Signal quality: 5-fold CV over all queries (analysis only; for routing, the probe's
    #    configuration is chosen on each training split alone).
    auroc = state.get("auroc") or {"V1 answer confidence": by_task(sig.slm_correct, sig.v1, sig.task),
                                   "V2 self-consistency, k=8": by_task(sig.slm_correct, sig.v2.mean(1), sig.task)}
    for pool in (POOLS + ("prompt",)) if "auroc" not in state else ():
        for layer in range(sig.hidden[pool].shape[1]):
            x = sig.hidden[pool][:, layer].astype(np.float32) if pool == "prompt" else features(sig, pool, layer)
            name = (f"Probe on prompt state (query only), layer {layer + 1}/4" if pool == "prompt"
                    else f"Probe, {pool} token{'s' if pool == 'mean' else ''} + V1, layer {layer + 1}/4")
            auroc[name] = by_task(sig.slm_correct, _cv_auroc(x, sig.slm_correct), sig.task)
            print(name, auroc[name], flush=True)

    # 2) Routing
    res = {r: state.get("res", {}).get(str(r), {}) for r in a.ratios}
    configs = state.get("configs", [])
    checkpoint()
    for sp in range(len(configs), a.splits):
        t0 = time.time()
        tr_i, te_i = split(len(sig), 0.3, sp)
        tr, te = sig.subset(tr_i), sig.subset(te_i)
        trp, tep, pr = with_probe(tr, te, seed=sp)
        ptrp, ptep, ppr = with_probe(pseudo_labeled(tr), te, seed=sp)
        # V0 from the SLM's prompt state: layer chosen by CV on the training split
        aucs = [roc_auc_score(tr.slm_correct, _cv_auroc(tr.hidden["prompt"][:, layer].astype(np.float32),
                                                        tr.slm_correct, seed=sp))
                for layer in range(tr.hidden["prompt"].shape[1])]
        prior_layer = int(np.argmax(aucs))
        trps, teps = slm_prior(trp, tep, prior_layer)
        configs.append({"probe": pr.config, "probe_label_free": ppr.config, "prior_layer": prior_layer,
                        "test_auroc": by_task(te.slm_correct, tep.v1, te.task),
                        "test_auroc_label_free": by_task(te.slm_correct, ptep.v1, te.task)})
        m_base, m_probe = BeliefModel(4, seed=0).fit(tr), BeliefModel(4, seed=0).fit(trp)
        m_free, m_prior = BeliefModel(4, seed=0).fit(ptrp), BeliefModel(4, seed=0).fit(trps)
        # probe trained on LLM agreement, router on 50 gold labels + calibrated agreement
        gold = np.isin(tr_i, np.random.default_rng(100 + sp).permutation(tr_i)[:50])
        soft, w = soft_labeled(replace(tr, v1=ptrp.v1), gold, seed=0)
        m_cal = BeliefModel(4, seed=0).fit(soft, w)
        # Reading the SLM's prompt state means running its prefill even for skipped queries:
        # charged as the prompt's share of the SLM call (input priced like output: conservative)
        share = te.tokens["slm_in"] / (te.tokens["slm_in"] + te.tokens["slm_out"])
        for r in a.ratios:
            cm = _cost(sig, te_i, r, a.cost_model)
            lams = routers.lambda_grid(cm, 24)
            prefill = share * cm.per_query["gen"]
            row = {
                "Ours (V1 log-prob), as in the main results": [routers.greedy_voi(m_base, te, cm, l) for l in lams],
                "Ours + probe": [routers.greedy_voi(m_probe, tep, cm, l) for l in lams],
                "Ours + probe, free signals only (no paid verifier)":
                    [routers.greedy_voi(m_probe, tep, cm, l, menu=()) for l in lams],
                "Ours + probe + SLM-state prior (prefill charged)":
                    [routers.greedy_voi(m_prior, teps, cm, l, skip_extra=prefill) for l in lams],
                "Ours + probe, no gold labels": [routers.greedy_voi(m_free, ptep, cm, l) for l in lams],
                "Ours + probe, 50 gold labels + calibrated agreement":
                    [routers.greedy_voi(m_cal, ptep, cm, l) for l in lams],
                "Fixed V1 (log-prob threshold)": routers.fixed_verifier("v1", te, cm),
                "Fixed probe threshold": routers.threshold_sweep(tep.v1, te, cm, np.zeros(len(te))),
                "Pre-generation router (V0 sentence embedding)": routers.pregen_router(m_base, te, cm),
                "Pre-generation router (SLM prompt state, prefill charged)":
                    routers.pregen_router(m_prior, teps, cm, skip_extra=prefill),
            }
            if a.cost_model == "token":
                from .costs import CostPredictor
                est = CostPredictor().fit(tr.emb, tr.tokens["slm_out"], _cost(sig, tr_i, r, a.cost_model)) \
                    .predict(te.emb, te.tokens["slm_out"])
                row["Ours, per-query costs"] = [routers.greedy_voi(m_base, te, cm, l, cost_estimates=est)
                                                for l in lams]
                row["Ours + probe, per-query costs"] = [routers.greedy_voi(m_probe, tep, cm, l, cost_estimates=est)
                                                        for l in lams]
                row["Ours + probe + SLM-state prior, per-query costs (prefill charged)"] = [
                    routers.greedy_voi(m_prior, teps, cm, l, cost_estimates=est, skip_extra=prefill) for l in lams]
            for name, o in row.items():
                res[r].setdefault(name, []).append(_summ(o, te, cm))
        checkpoint()
        print(f"probe: split {sp} {time.time() - t0:.0f}s  {configs[-1]}", flush=True)
    md = _table(f"Hidden-state probe as the free signal ({a.cost_model} costs)",
                "The probe (logistic regression on one layer of the SLM's hidden states plus V1) replaces V1. "
                "Pooling, layer and regularisation are chosen by 5-fold CV on each training split. "
                f"Mean ± std over {a.splits} splits.", res, a.ratios)
    ta = np.array([[c["test_auroc"][t] for t in tasks] for c in configs])
    tf = np.array([[c["test_auroc_label_free"][t] for t in tasks] for c in configs])
    md += ["", "## Signal AUROC for SLM correctness (5-fold CV over all queries, C = 0.01)", "",
           "| Signal | Math | Factual |", "| --- | --- | --- |"]
    md += [f"| {n} | {v['math']:.3f} | {v['factual']:.3f} |" for n, v in auroc.items()]
    md += ["", f"Probe chosen on each training split, AUROC on its test split (mean over splits): "
               f"math {ta[:, 0].mean():.3f}, factual {ta[:, 1].mean():.3f}. Trained on LLM agreement instead "
               f"of gold labels: math {tf[:, 0].mean():.3f}, factual {tf[:, 1].mean():.3f}.",
           "", "Chosen configurations: " + "; ".join(
               f"split {i}: {c['probe']['pool']}, layer {c['probe']['layer'] + 1}/4, C={c['probe']['C']:g}"
               for i, c in enumerate(configs))]
    _save(out, f"probe_{a.cost_model}", {"auroc": auroc, "configs": configs, "results": res}, "\n".join(md) + "\n")
    partial.unlink()


CMDS = {"signals": cmd_signals, "exact": cmd_exact, "bootstrap": cmd_bootstrap, "ablate": cmd_ablate,
        "trainsize": cmd_trainsize, "shift": cmd_shift, "labelfree": cmd_labelfree, "probe": cmd_probe}


def main(argv=None):
    import sys
    sys.stdout.reconfigure(encoding="utf-8")  # tables contain non-ASCII; Windows consoles default to cp1252
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cmd", choices=CMDS)
    p.add_argument("--cache", default="cache")
    p.add_argument("--out", default="results/ablations")
    p.add_argument("--ratios", default="5,20,60")
    p.add_argument("--splits", type=int, default=5)
    p.add_argument("--n", type=int, default=400, help="test subsample size for `exact`")
    p.add_argument("--boot", type=int, default=1000)
    p.add_argument("--cost-model", choices=["unit", "token"], default="unit", help="for labelfree and probe")
    a = p.parse_args(argv)
    a.ratios = [int(x) for x in a.ratios.split(",")]
    sig = load(a.cache)
    CMDS[a.cmd](sig, Path(a.out), a)


if __name__ == "__main__":
    main()
