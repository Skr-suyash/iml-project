"""Figures for the report, from the saved results (plus one re-run of the router for action shares).

    python scripts/make_figures.py      # writes report/figures/*.png
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vcascade import routers, signals  # noqa: E402
from vcascade.belief import BeliefModel  # noqa: E402
from vcascade.costs import CostModel  # noqa: E402
from vcascade.experiment import METHOD_LABELS, split  # noqa: E402

FIG = ROOT / "report" / "figures"
plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3, "axes.spines.top": False,
                     "axes.spines.right": False})


def copy_main():
    for name in ("curves_r5.png", "curves_r20.png", "curves_r60.png", "lift_vs_ratio.png"):
        shutil.copy(ROOT / "results" / "main" / name, FIG / f"main_{name}")


def signal_auroc():
    d = json.loads((ROOT / "results" / "ablations" / "signals.json").read_text(encoding="utf-8"))["auroc"]
    names = list(d)
    short = ["V0 prior", "V1 conf.", "V2 k=1", "V2 k=2", "V2 k=4", "V2 k=8", "V3 retr.", "V4 no-ctx"]
    cost = [0, 0, 1, 2, 4, 8, 1.5, 1]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.bar(x - 0.2, [d[n]["math"] for n in names], 0.4, label="Math (GSM8K)", color="C0")
    ax.bar(x + 0.2, [d[n]["factual"] for n in names], 0.4, label="Factual (TriviaQA)", color="C1")
    ax.set_xticks(x, [f"{s}\ncost {c:g}" for s, c in zip(short, cost)], fontsize=8)
    ax.set_ylim(0.5, 0.92)
    ax.set_ylabel("AUROC for SLM correctness")
    ax.set_title("Which signal predicts a wrong SLM answer, by task")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "signal_auroc.png", dpi=160)
    plt.close(fig)


def actions_by_task():
    sig = signals.load(ROOT / "cache")
    tr_i, te_i = split(len(sig), 0.3, 0)
    tr, te = sig.subset(tr_i), sig.subset(te_i)
    m = BeliefModel(4).fit(tr)
    ratio = 60
    cm = CostModel.unit(len(te), ratio)
    lcs = np.geomspace(0.01, 0.6, 14)
    rows = {t: {k: [] for k in ("skip", "v2", "v3", "v4", "escalate")} for t in ("math", "factual")}
    for lc in lcs:
        o = routers.greedy_voi(m, te, cm, lc / ratio)
        for t in rows:
            k = te.task == t
            for a in rows[t]:
                rows[t][a].append(o.actions[a][k].mean())
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    labels = {"skip": "skip SLM, go to LLM", "v2": "V2 samples per query", "v3": "V3 retrieval verify",
              "v4": "V4 context-free verify", "escalate": "escalate (total)"}
    for ax, t in zip(axes, rows):
        for a, c in zip(labels, ["C5", "C0", "C1", "C2", "k"]):
            ax.plot(lcs, rows[t][a], label=labels[a], color=c, ls="--" if a == "escalate" else "-")
        ax.set_xscale("log")
        ax.set_title(f"{t.capitalize()} questions")
        ax.set_xlabel("λ × C_LLM (higher = tighter budget)")
    axes[0].set_ylabel("Share of queries (V2: mean samples)")
    axes[1].legend(fontsize=7, loc="upper right")
    fig.suptitle("What the router buys, by task (task label never seen; cost ratio 60, split 0)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "actions_by_task.png", dpi=160)
    plt.close(fig)


def trainsize():
    d = json.loads((ROOT / "results" / "ablations" / "trainsize.json").read_text(encoding="utf-8"))
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    sizes = ["50", "100", "200", "400", "all"]
    xs = [50, 100, 200, 400, 828]
    for ax, r in zip(axes, d):
        for m, c in zip(["greedy_voi", "fixed_v1", "category_map", "pregen", "automix"],
                        ["C3", "C9", "C4", "C5", "C6"]):
            mean = [np.mean([x["area_lift"] for x in d[r][s][m]]) for s in sizes]
            sd = [np.std([x["area_lift"] for x in d[r][s][m]]) for s in sizes]
            ax.errorbar(xs, mean, yerr=sd, label=METHOD_LABELS[m], color=c, lw=2.2 if m == "greedy_voi" else 1.2,
                        capsize=2)
        ax.set_xscale("log")
        ax.set_xticks(xs, [str(x) for x in xs])
        ax.minorticks_off()
        ax.set_title(f"Cost ratio {r}")
        ax.set_xlabel("Training examples")
    axes[0].set_ylabel("Area lift (accuracy points)")
    axes[1].legend(fontsize=7, loc="lower right")
    fig.suptitle("How much labelled data the routers need", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "trainsize.png", dpi=160)
    plt.close(fig)


def label_efficiency():
    """Area lift vs number of gold labels: gold only vs calibrated LLM agreement (unit costs)."""
    d = json.loads((ROOT / "results" / "ablations" / "labelfree_unit.json").read_text(encoding="utf-8"))["results"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    for ax, r in zip(axes, ("20", "60")):
        def ms(name):
            v = [x["area_lift"] for x in d[r][name]]
            return np.mean(v), np.std(v)
        gold = [ms(f"Ours, gold labels ({n})") for n in (50, 100, 200)] + [ms("Ours, gold labels (all 828)")]
        cal = [ms(f"Ours, {n} gold labels + calibrated LLM agreement") for n in (20, 50, 100)]
        ax.errorbar([50, 100, 200, 828], [m for m, _ in gold], yerr=[s for _, s in gold], color="C3", marker="o",
                    capsize=2, label="Ours, gold labels only")
        ax.errorbar([20, 50, 100], [m for m, _ in cal], yerr=[s for _, s in cal], color="C2", marker="s", capsize=2,
                    lw=2.2, label="Ours, gold labels + calibrated LLM agreement")
        m0, _ = ms("Ours, no gold labels (LLM answer as truth)")
        v1, _ = ms("Fixed V1 (needs no labels)")
        ax.axhline(m0, color="C2", ls=":", lw=1.2, label="Ours, no gold labels (LLM answer as truth)")
        ax.axhline(v1, color="C9", ls="--", lw=1.2, label="Fixed V1 threshold (no labels)")
        ax.set_xscale("log")
        ax.set_xticks([20, 50, 100, 200, 828], ["20", "50", "100", "200", "828"])
        ax.minorticks_off()
        ax.set_title(f"Cost ratio {r}")
        ax.set_xlabel("Gold-labelled training queries")
    axes[0].set_ylabel("Area lift (accuracy points)")
    axes[1].legend(fontsize=7, loc="lower right")
    fig.suptitle("Label efficiency: the LLM's answers stand in for most gold labels (unit costs)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "label_efficiency.png", dpi=160)
    plt.close(fig)


def probe_gain():
    """Area lift per cost ratio: V1 router vs probe variants, unit and token costs."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
    for ax, cost in zip(axes, ("unit", "token")):
        d = json.loads((ROOT / "results" / "ablations" / f"probe_{cost}.json").read_text(encoding="utf-8"))["results"]
        ratios = list(d)
        names = [("Fixed V1 (log-prob threshold)", "C9", "--"), ("Ours (V1 log-prob), as in the main results", "C3", "-"),
                 ("Ours + probe", "C1", "-")]
        if cost == "unit":
            names += [("Ours + probe + SLM-state prior (prefill charged)", "C2", "-")]
        else:
            names += [("Ours, per-query costs", "C8", ":"), ("Ours + probe, per-query costs", "C4", "-"),
                      ("Ours + probe + SLM-state prior, per-query costs (prefill charged)", "C6", "-")]
        for name, c, ls in names:
            v = np.array([[x["area_lift"] for x in d[r][name]] for r in ratios])
            ax.errorbar([int(r) for r in ratios], v.mean(1), yerr=v.std(1), color=c, ls=ls, marker="o", capsize=2,
                        label=name.replace(", as in the main results", "").replace(" (prefill charged)", ""))
        ax.set_xscale("log")
        ax.set_xticks([int(r) for r in ratios], ratios)
        ax.minorticks_off()
        ax.set_title(f"{cost.capitalize()} costs")
        ax.set_xlabel("LLM/SLM cost ratio")
    axes[0].set_ylabel("Area lift (accuracy points)")
    handles = {}
    for ax in axes:
        for h, lab in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(lab, h)
    fig.legend(handles.values(), handles.keys(), fontsize=7, loc="lower center", ncol=3, frameon=False)
    fig.suptitle("Hidden-state probe as the free signal (SLM-state prior: prompt pass charged)", fontsize=10)
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    fig.savefig(FIG / "probe_gain.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    copy_main()
    signal_auroc()
    actions_by_task()
    trainsize()
    label_efficiency()
    probe_gain()
    print("figures in", FIG)
