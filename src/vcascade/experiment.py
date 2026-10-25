"""Run every router on the cached signals and report cost-accuracy results.

    python -m vcascade.experiment --cache cache --out results
    python -m vcascade.experiment --synthetic --out results/synthetic     # pipeline test

For each random train/test split and each LLM/SLM cost ratio, every method is
fitted on the train split and evaluated on the test split. Output:
    results.json   per-split, per-ratio summaries (frontiers, IBC lift, area lift)
    summary.md     mean +- std table per cost ratio
    curves_r<R>.png cost-accuracy curves for the first split
    lift_vs_ratio.png
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from . import evaluate, routers
from .belief import BeliefModel
from .costs import CostModel

METHOD_LABELS = {
    "fixed_v1": "Fixed V1 (answer confidence)",
    "fixed_v2": "Fixed V2 (self-consistency, best k)",
    "fixed_v3": "Fixed V3 (retrieval verify)",
    "fixed_v4": "Fixed V4 (context-free verify)",
    "category_map": "Per-category map (oracle task)",
    "pregen": "Pre-generation router (V0)",
    "automix": "AutoMix (V4, Bayes/POMDP)",
    "compute_all": "Compute everything",
    "greedy_voi": "Ours: greedy VOI",
    "greedy_voi_cc": "Ours: greedy VOI, per-query costs",
    "exact": "Ours: exact planner",
}


def split(n: int, train_frac: float, seed: int):
    idx = np.random.default_rng(seed).permutation(n)
    k = int(round(train_frac * n))
    return np.sort(idx[:k]), np.sort(idx[k:])


def run_methods(tr, te, cm, args, cm_tr=None) -> dict:
    outs, slm, llm = compute_outcomes(tr, te, cm, args, cm_tr)
    res = {"slm": slm, "llm": llm, "methods": {}, "args_exact_n": args.exact_n}
    for name, o in outs.items():
        if name in ("exact", "greedy_on_exact_subset"):
            continue
        res["methods"][name] = evaluate.summarize(o, slm, llm)
        res["methods"][name]["info"] = [x.info for x in o]
    if args.exact:
        sub = np.random.default_rng(args.seed).permutation(len(te))[:args.exact_n]
        te_s, cm_s = te.subset(sub), cm.subset(sub)
        ref_s, ref_l = routers.always_slm(te_s, cm_s).point, routers.always_llm(te_s, cm_s).point
        for name in ("exact", "greedy_on_exact_subset"):
            res["methods"][name] = evaluate.summarize(outs[name], ref_s, ref_l)
            res["methods"][name]["info"] = [x.info for x in outs[name]]
    return res


def compute_outcomes(tr, te, cm, args, cm_tr=None):
    """Every method's operating points (per-query outcomes) on one split and cost model.

    With cm_tr (train-split costs) and costs that vary across queries, also runs the
    router with per-query cost estimates fitted on the train split.
    """
    model = BeliefModel(n_groups=args.groups, seed=args.seed).fit(tr)
    automix_model = BeliefModel(n_groups=1, seed=args.seed).fit(tr)
    slm = routers.always_slm(te, cm).point
    llm = routers.always_llm(te, cm).point
    outs = {
        "fixed_v1": routers.fixed_verifier("v1", te, cm),
        "fixed_v2": routers.fixed_verifier("v2", te, cm),
        "fixed_v3": routers.fixed_verifier("v3", te, cm),
        "fixed_v4": routers.fixed_verifier("v4", te, cm),
        "category_map": routers.category_map(tr, te, cm),
        "pregen": routers.pregen_router(model, te, cm),
        "automix": routers.posterior_gain_sweep(automix_model, te, cm, ("v4",), use_prior=False),
        "compute_all": routers.posterior_gain_sweep(model, te, cm, ("v1", "v2", "v3", "v4")),
    }
    lams = routers.lambda_grid(cm, args.n_lambda)
    outs["greedy_voi"] = [routers.greedy_voi(model, te, cm, lam) for lam in lams]
    if cm_tr is not None and np.std(cm.per_query["gen"]) > 0:
        from .costs import CostPredictor
        est = CostPredictor().fit(tr.emb, tr.tokens["slm_out"], cm_tr).predict(te.emb, te.tokens["slm_out"])
        outs["greedy_voi_cc"] = [routers.greedy_voi(model, te, cm, lam, cost_estimates=est) for lam in lams]
    if args.exact:
        sub = np.random.default_rng(args.seed).permutation(len(te))[:args.exact_n]
        te_s, cm_s = te.subset(sub), cm.subset(sub)
        outs["exact"] = [routers.exact_planner(model, te_s, cm_s, lam) for lam in lams[::2]]
        outs["greedy_on_exact_subset"] = [routers.greedy_voi(model, te_s, cm_s, lam) for lam in lams[::2]]
    return outs, slm, llm


def write_summary(all_res, ratios, out: Path) -> str:
    lines = ["# Results", ""]
    for r in ratios:
        lines += [f"## Cost ratio LLM/SLM = {r}", "",
                  "| Method | IBC lift (%) | Area lift (acc. points) | Max accuracy |",
                  "| --- | --- | --- | --- |"]
        per = [s[str(r)] for s in all_res]

        def row(label, m):
            ibc = [p["methods"][m]["ibc_lift"] for p in per]
            area = [p["methods"][m]["area_lift"] for p in per]
            mx = [p["methods"][m]["max_perf"] for p in per]
            return (f"| {label} | {np.mean(ibc):.1f} ± {np.std(ibc):.1f} "
                    f"| {np.mean(area):.2f} ± {np.std(area):.2f} | {np.mean(mx):.3f} |")

        subset_only = ("exact", "greedy_on_exact_subset")
        names = [m for m in per[0]["methods"] if m not in subset_only]
        for m in sorted(names, key=lambda m: -np.mean([p["methods"][m]["area_lift"] for p in per])):
            lines.append(row(METHOD_LABELS.get(m, m), m))
        s0 = per[0]
        lines += ["", f"Always-SLM: cost {s0['slm'][0]:.2f}, acc {s0['slm'][1]:.3f}. "
                      f"Always-LLM: cost {s0['llm'][0]:.2f}, acc {s0['llm'][1]:.3f} (split 0).", ""]
        if "exact" in per[0]["methods"]:
            n_sub = per[0]["args_exact_n"] if "args_exact_n" in per[0] else "a"
            lines += [f"Exact planner vs. greedy on the same {n_sub}-query test subset:", "",
                      "| Method | IBC lift (%) | Area lift (acc. points) | Max accuracy |",
                      "| --- | --- | --- | --- |",
                      row("Ours: exact planner (max 4 samples)", "exact"),
                      row("Ours: greedy VOI", "greedy_on_exact_subset"), ""]
    text = "\n".join(lines)
    (out / "summary.md").write_text(text, encoding="utf-8")
    return text


def plot(all_res, ratios, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    show = ["fixed_v1", "fixed_v2", "fixed_v3", "fixed_v4", "category_map", "pregen", "automix", "compute_all",
            "greedy_voi", "greedy_voi_cc"]
    colors = dict(zip(show, ["C9", "C0", "C1", "C2", "C4", "C5", "C6", "C7", "C3", "C8"]))  # ours in red / olive
    show = [m for m in show if m in all_res[0][str(ratios[0])]["methods"]]
    for r in ratios:
        res = all_res[0][str(r)]
        fig, ax = plt.subplots(figsize=(7, 5))
        (cs, ps), (cl, pl) = res["slm"], res["llm"]
        ax.plot([cs, cl], [ps, pl], "k--", lw=1, label="Random mixing")
        for m in show:
            fr = np.array(res["methods"][m]["frontier"])
            style = dict(lw=2.5 if m.startswith("greedy") else 1.2, color=colors[m])
            ax.plot(fr[:, 0], fr[:, 1], label=METHOD_LABELS[m], **style)
        ax.set_xlim(cs * 0.9, cl * 1.05)
        ax.set_xlabel("Average cost per query (SLM-call units)")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"Cost-accuracy frontier, LLM/SLM cost ratio {r} (split 0)")
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out / f"curves_r{r}.png", dpi=150)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for m in show:
        vals = [np.mean([s[str(r)]["methods"][m]["ibc_lift"] for s in all_res]) for r in ratios]
        style = dict(lw=2.5 if m.startswith("greedy") else 1.2, color=colors[m])
        ax.plot(ratios, vals, marker="o", label=METHOD_LABELS[m], **style)
    ax.set_xscale("log")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("LLM/SLM cost ratio")
    ax.set_ylabel("IBC lift (%)")
    ax.set_title("IBC lift vs. cost ratio (mean over splits)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "lift_vs_ratio.png", dpi=150)
    plt.close(fig)


def main(argv=None):
    import sys
    sys.stdout.reconfigure(encoding="utf-8")  # tables contain non-ASCII; Windows consoles default to cp1252
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache", default="cache")
    p.add_argument("--synthetic", action="store_true", help="use simulated signals instead of --cache")
    p.add_argument("--out", default="results")
    p.add_argument("--ratios", default="3,5,10,20,60,200")
    p.add_argument("--cost-model", choices=["unit", "token"], default="unit")
    p.add_argument("--splits", type=int, default=5)
    p.add_argument("--train-frac", type=float, default=0.3)
    p.add_argument("--groups", type=int, default=4, help="k-means query groups (1 = none)")
    p.add_argument("--n-lambda", type=int, default=24)
    p.add_argument("--exact", action="store_true", help="also run the exact planner on a subsample")
    p.add_argument("--exact-n", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    if args.synthetic:
        from .simulate import make_signals
        sig = make_signals(seed=args.seed)
    else:
        from .signals import load
        sig = load(args.cache)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ratios = [float(r) if "." in r else int(r) for r in args.ratios.split(",")]

    all_res = []
    for sp in range(args.splits):
        tr_idx, te_idx = split(len(sig), args.train_frac, seed=args.seed + sp)
        tr, te = sig.subset(tr_idx), sig.subset(te_idx)
        per_ratio = {}
        for r in ratios:
            t0 = time.time()
            if args.cost_model == "unit":
                cm_all = CostModel.unit(len(sig), ratio=r)
            else:  # scale shared by train and test: mean SLM answer over all queries = 1
                cm_all = CostModel.token(sig.tokens, llm_price=(r, r))
            cm, cm_tr = cm_all.subset(te_idx), cm_all.subset(tr_idx)
            per_ratio[str(r)] = run_methods(tr, te, cm, args, cm_tr)
            print(f"split {sp} ratio {r}: done in {time.time() - t0:.1f}s")
        all_res.append(per_ratio)

    (out / "results.json").write_text(json.dumps({"args": vars(args), "splits": all_res}), encoding="utf-8")
    print(write_summary(all_res, ratios, out))
    plot(all_res, ratios, out)


if __name__ == "__main__":
    main()
