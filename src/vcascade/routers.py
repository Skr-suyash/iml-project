"""Routing policies.

Every policy returns, per query, the accuracy it achieved and the cost it paid.
Cost-quality curves come from sweeping either a threshold (baselines whose
decision is one score against a cut-off) or lambda, the price of one unit of
cost in accuracy, for routers whose acquisition decisions depend on it.

Decision rule shared by the belief-based routers, at belief b:
    accept   value = P(SLM correct)            = b[easy] + b[slm_only]
    escalate value = P(LLM correct) - lam * C_llm = b[easy] + b[hard] - lam * C_llm
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

from .belief import NZ, BeliefModel, normalize, value_accept, value_escalate
from .costs import CostModel
from .signals import Signals


@dataclass
class Outcome:
    perf: np.ndarray                  # (n,) accuracy per query
    cost: np.ndarray                  # (n,) cost paid per query
    info: dict = field(default_factory=dict)
    actions: dict = field(default_factory=dict)  # per-query action arrays, for analysis (not saved)

    @property
    def point(self) -> tuple[float, float]:
        return float(self.cost.mean()), float(self.perf.mean())


# ---------------------------------------------------------------- baselines

def always_slm(sig: Signals, cm: CostModel) -> Outcome:
    return Outcome(sig.slm_correct.astype(float), cm.per_query["gen"].copy())


def always_llm(sig: Signals, cm: CostModel) -> Outcome:
    return Outcome(sig.llm_correct.astype(float), cm.per_query["llm"].copy())


def threshold_sweep(score: np.ndarray, sig: Signals, cm: CostModel, extra_cost: np.ndarray,
                    generate: np.ndarray | None = None) -> list[Outcome]:
    """Escalate when score < t, for every cut-off t that changes a decision.

    `generate` (bool per query, default all True): whether the SLM was called
    before the decision. False only for pre-generation routing.
    """
    gen = np.ones(len(score), bool) if generate is None else generate
    cuts = np.concatenate([np.unique(score), [np.inf]])
    outs = []
    for t in cuts:
        esc = score < t
        perf = np.where(esc, sig.llm_correct, sig.slm_correct).astype(float)
        cost = gen * (cm.per_query["gen"] + extra_cost) + esc * cm.per_query["llm"]
        outs.append(Outcome(perf, cost, {"threshold": float(t), "escalation_rate": float(esc.mean())}))
    return outs


V2_KS = (1, 2, 4, 8)


def fixed_verifier(name: str, sig: Signals, cm: CostModel, k: int | None = None) -> list[Outcome]:
    """AutoMix-threshold-style cascade with one fixed verifier.

    For V2 without an explicit k, every sample count in V2_KS is swept and all their
    points are returned, so the baseline's frontier uses its best k at each budget.
    """
    if name == "v1":
        return threshold_sweep(sig.v1, sig, cm, np.zeros(len(sig)))
    if name == "v2":
        ks = [k] if k else [kk for kk in V2_KS if kk <= sig.v2.shape[1]]
        return [o for kk in ks
                for o in threshold_sweep(sig.v2[:, :kk].mean(1), sig, cm, kk * cm.per_query["sample"])]
    if name in ("v3", "v4"):
        return threshold_sweep(getattr(sig, name).astype(float), sig, cm, cm.per_query[name])
    raise ValueError(name)


def category_map(sig_tr: Signals, sig: Signals, cm: CostModel, mapping=None) -> list[Outcome]:
    """Hand-coded verifier per task, given the TRUE task label (an oracle advantage).

    Scores are calibrated per task to P(SLM correct | score) on the training split,
    then a single cut-off on that probability is swept. V2's sample count is swept
    over V2_KS and all points returned, as for the fixed-V2 baseline.
    """
    mapping = mapping or {"math": "v2", "factual": "v3"}

    def raw(s: Signals, name, k):
        return s.v2[:, :k].mean(1) if name == "v2" else getattr(s, name).astype(float)

    outs = []
    for k in [kk for kk in V2_KS if kk <= sig.v2.shape[1]]:
        p = np.zeros(len(sig))
        extra = np.zeros(len(sig))
        for task, name in mapping.items():
            tr, te = sig_tr.task == task, sig.task == task
            x_tr, x_te = raw(sig_tr, name, k)[tr], raw(sig, name, k)[te]
            y_tr = sig_tr.slm_correct[tr]
            levels = np.unique(x_tr)
            rate = {lv: (y_tr[x_tr == lv].sum() + 1) / ((x_tr == lv).sum() + 2) for lv in levels}  # Laplace
            base = (y_tr.sum() + 1) / (len(y_tr) + 2)
            p[te] = [rate.get(v, base) for v in x_te]
            extra[te] = k * cm.per_query["sample"][te] if name == "v2" else cm.per_query[name][te]
        outs += threshold_sweep(p, sig, cm, extra)
    return outs


def learned_combiner(sig_tr: Signals, sig: Signals, cm: CostModel) -> list[Outcome]:
    """'Compute everything' with a learned combiner instead of naive Bayes.

    Logistic regressions on all four signals predict P(SLM correct) and P(LLM correct);
    escalate when the predicted gain is large. Pays for every signal on every query.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    def feats(s):
        return np.column_stack([s.v1, s.v2.mean(1), s.v3, s.v4])

    sc = StandardScaler().fit(feats(sig_tr))
    x_tr, x = sc.transform(feats(sig_tr)), sc.transform(feats(sig))
    p_s = LogisticRegression().fit(x_tr, sig_tr.slm_correct).predict_proba(x)[:, 1]
    p_l = LogisticRegression().fit(x_tr, sig_tr.llm_correct).predict_proba(x)[:, 1]
    extra = sig.v2.shape[1] * cm.per_query["sample"] + cm.per_query["v3"] + cm.per_query["v4"]
    return threshold_sweep(p_s - p_l, sig, cm, extra)


def pregen_router(model: BeliefModel, sig: Signals, cm: CostModel,
                  skip_extra: np.ndarray | None = None) -> list[Outcome]:
    """Decide from the query alone (HybridLLM / RouteLLM style): escalate before generating.

    skip_extra (n,): cost an escalated query still pays, e.g. the SLM prefill when the prior
    reads the SLM's own prompt state.
    """
    extra = np.zeros(len(sig)) if skip_extra is None else skip_extra
    b0 = model.prior(sig.emb)
    score = value_accept(b0) - value_escalate(b0)  # escalate when the LLM's expected gain is large
    outs = []
    for t in np.concatenate([np.unique(score), [np.inf]]):
        esc = score < t
        perf = np.where(esc, sig.llm_correct, sig.slm_correct).astype(float)
        cost = np.where(esc, cm.per_query["llm"] + extra, cm.per_query["gen"])  # escalated queries skip the SLM
        outs.append(Outcome(perf, cost, {"threshold": float(t), "escalation_rate": float(esc.mean())}))
    return outs


def posterior_gain_sweep(model: BeliefModel, sig: Signals, cm: CostModel, evidence: tuple[str, ...],
                         use_prior: bool = True) -> list[Outcome]:
    """Gather a fixed set of signals for every query, then escalate if the posterior gain is high.

    evidence=("v4",), use_prior=False with a 1-group model is AutoMix's router for two
    models: one observation, one decision (the POMDP reduces to this Bayes rule).
    evidence=("v1","v2","v3","v4") is the 'compute everything' reference.
    """
    n = len(sig)
    g = model.group(sig.emb)
    b = model.prior(sig.emb) if use_prior else np.tile(model.freq, (n, 1))
    extra = np.zeros(n)
    K = sig.v2.shape[1]
    for i in range(n):
        lik = np.ones(NZ)
        if "v1" in evidence:
            lik *= model.v1_lik(float(sig.v1[i]), g[i])
        if "v2" in evidence:
            lik *= model.v2.lik(sig.v2[i].sum(), K, g[i])
        if "v3" in evidence:
            lik *= model.v3.lik(sig.v3[i], g[i])
        if "v4" in evidence:
            lik *= model.v4.lik(sig.v4[i], g[i])
        b[i] = b[i] * lik
    b = normalize(b)
    if "v2" in evidence:
        extra += K * cm.per_query["sample"]
    for name in ("v3", "v4"):
        if name in evidence:
            extra += cm.per_query[name]
    gain = value_escalate(b) - value_accept(b)
    return threshold_sweep(-gain, sig, cm, extra)


# ---------------------------------------------------- value-of-information routers

class _Query:
    """One query's view for the sequential routers: priors, group, and its cached signals."""

    def __init__(self, model: BeliefModel, sig: Signals, i: int, b0: np.ndarray, g: int):
        self.m, self.i, self.b0, self.g = model, i, b0, g
        self.v1 = float(sig.v1[i])
        self.v2 = sig.v2[i]
        self.v3 = int(sig.v3[i])
        self.v4 = int(sig.v4[i])


def _act_value(b, lam, c_llm):
    return max(value_accept(b), value_escalate(b) - lam * c_llm)


def _expected_act_value(b, P, lam, c_llm):
    """E over an observation with likelihood table P (NZ, L) of the best act value afterwards."""
    joint = b[:, None] * P                     # (NZ, L)
    pred = joint.sum(0)                        # (L,)
    post = joint / np.maximum(pred, 1e-300)    # columns are posteriors
    acc = post[0] + post[3]
    esc = post[0] + post[1] - lam * c_llm
    return float(np.sum(pred * np.maximum(acc, esc)))


def greedy_voi(model: BeliefModel, sig: Signals, cm: CostModel, lam: float,
               v2_batches=(1, 2, 4, 8), allow_skip: bool = True,
               menu=("v2", "v3", "v4"), use_prior: bool = True, use_v1: bool = True,
               cost_estimates: tuple[dict, dict] | None = None,
               skip_extra: np.ndarray | None = None) -> Outcome:
    """Myopic value-of-information router.

    Before generating: escalate directly if that beats generating and then acting.
    After generating (V1 comes free): repeatedly buy the signal whose expected
    improvement in the final decision most exceeds lam * its cost; stop when none does.
    For V2, lookahead considers batches of m more samples, so that a signal that
    only pays off after a few draws is not dismissed, but samples are drawn one at a time.

    cost_estimates = (before generation, after generation), each action -> (n,) predicted
    per-query costs (see CostPredictor). Without it the router plans with mean costs.

    skip_extra (n,): cost a skipped query still pays (the SLM prefill, when the prior reads
    the SLM's prompt state); planned with its mean, charged per query.
    """
    K = sig.v2.shape[1]
    n = len(sig)
    if cost_estimates is None:
        pre = post = {a: np.full(n, c) for a, c in cm.mean.items()}
    else:
        pre, post = cost_estimates
    b0s = model.prior(sig.emb) if use_prior else np.tile(model.freq, (n, 1))
    gs = model.group(sig.emb)
    extra = np.zeros(n) if skip_extra is None else skip_extra
    extra_plan = float(extra.mean())
    perf, cost = np.zeros(n), np.zeros(n)
    counts = {"skip": np.zeros(n), "v2": np.zeros(n), "v3": np.zeros(n), "v4": np.zeros(n),
              "escalate": np.zeros(n)}
    for i in range(n):
        q = _Query(model, sig, i, b0s[i], gs[i])
        b = q.b0
        C = {a: v[i] for a, v in pre.items()}
        if allow_skip:
            v_esc = value_escalate(b) - lam * (C["llm"] + extra_plan)
            if use_v1:
                v_after = _expected_act_value(b, model.v1.table[q.g], lam, C["llm"])
            else:
                v_after = _act_value(b, lam, C["llm"])
            v_gen = v_after - lam * C["gen"]
            if v_esc >= v_gen:
                perf[i], cost[i] = sig.llm_correct[i], cm.per_query["llm"][i] + extra[i]
                counts["skip"][i] = counts["escalate"][i] = 1
                continue
        cost[i] += cm.per_query["gen"][i]
        C = {a: v[i] for a, v in post.items()}
        if use_v1:
            b = normalize(b * model.v1_lik(q.v1, q.g))
        s = k = 0
        used = set()
        while True:
            v_now = _act_value(b, lam, C["llm"])
            best, best_gain = None, 1e-9
            for name in ("v3", "v4"):
                if name in menu and name not in used:
                    gain = _expected_act_value(b, getattr(model, name).table[q.g], lam, C["llm"]) \
                        - v_now - lam * C[name]
                    if gain > best_gain:
                        best, best_gain = name, gain
            if "v2" in menu and k < K:
                for m in sorted({min(mm, K - k) for mm in v2_batches}):
                    P = model.v2.predictive(s, k, m, q.g)
                    gain = _expected_act_value(b, P, lam, C["llm"]) - v_now - lam * m * C["sample"]
                    if gain > best_gain:
                        best, best_gain = "v2", gain
            if best is None:
                break
            if best == "v2":
                a = int(q.v2[k])
                b = normalize(b * model.v2.predictive(s, k, 1, q.g)[:, a])
                s, k = s + a, k + 1
                cost[i] += cm.per_query["sample"][i]
                counts["v2"][i] += 1
            else:
                b = normalize(b * getattr(model, best).lik(getattr(q, best), q.g))
                cost[i] += cm.per_query[best][i]
                counts[best][i] += 1
                used.add(best)
        if value_escalate(b) - lam * C["llm"] > value_accept(b):
            perf[i] = sig.llm_correct[i]
            cost[i] += cm.per_query["llm"][i]
            counts["escalate"][i] = 1
        else:
            perf[i] = sig.slm_correct[i]
    info = {"lam": lam, "skip_rate": float(counts["skip"].mean()),
            "mean_v2_samples": float(counts["v2"].mean()),
            "v3_rate": float(counts["v3"].mean()), "v4_rate": float(counts["v4"].mean()),
            "escalation_rate": float(counts["escalate"].mean())}
    return Outcome(perf, cost, info, actions=counts)


def exact_planner(model: BeliefModel, sig: Signals, cm: CostModel, lam: float, max_k: int = 4,
                  menu=("v2", "v3", "v4")) -> Outcome:
    """Non-myopic router: finite-horizon POMDP solved exactly by expectimax.

    State after generation = (s, k, V3 level or None, V4 level or None). With four
    latent types and at most max_k + 2 acquisitions the tree is small enough to
    search exhaustively for every query.
    """
    C = cm.mean
    b0s, gs = model.prior(sig.emb), model.group(sig.emb)
    n = len(sig)
    perf, cost = np.zeros(n), np.zeros(n)
    n_skip = 0
    acq = np.zeros(n)
    L3 = model.v3.table.shape[-1]

    for i in range(n):
        g = gs[i]
        T3, T4 = model.v3.table[g], model.v4.table[g]   # (NZ, L)

        def make_V(b_gen):
            @lru_cache(maxsize=None)
            def V(s, k, o3, o4):
                lik = model.v2.lik(s, k, g)
                if o3 is not None:
                    lik = lik * T3[:, o3]
                if o4 is not None:
                    lik = lik * T4[:, o4]
                b = normalize(b_gen * lik)
                best, best_a = _act_value(b, lam, C["llm"]), "act"
                if "v2" in menu and k < max_k:
                    p1 = model.v2.predictive(s, k, 1, g)          # (NZ, 2)
                    pred = b @ p1
                    val = -lam * C["sample"] + pred[0] * V(s, k + 1, o3, o4)[0] + pred[1] * V(s + 1, k + 1, o3, o4)[0]
                    if val > best:
                        best, best_a = val, "v2"
                for name, T, done in (("v3", T3, o3 is not None), ("v4", T4, o4 is not None)):
                    if name in menu and not done:
                        pred = b @ T
                        val = -lam * C[name] + sum(
                            pred[o] * V(s, k, o if name == "v3" else o3, o if name == "v4" else o4)[0]
                            for o in range(L3))
                        if val > best:
                            best, best_a = val, name
                return best, best_a
            return V

        b0 = b0s[i]
        P1 = model.v1.table[g]
        joint = b0[:, None] * P1
        pred1 = joint.sum(0)
        v_gen = -lam * C["gen"] + sum(
            pred1[l] * make_V(normalize(joint[:, l]))(0, 0, None, None)[0] for l in range(P1.shape[1]))
        if value_escalate(b0) - lam * C["llm"] >= v_gen:
            perf[i], cost[i] = sig.llm_correct[i], cm.per_query["llm"][i]
            n_skip += 1
            continue
        q = _Query(model, sig, i, b0, g)
        b_gen = normalize(b0 * model.v1_lik(q.v1, g))
        V = make_V(b_gen)
        cost[i] += cm.per_query["gen"][i]
        s = k = 0
        o3 = o4 = None
        while True:
            _, a = V(s, k, o3, o4)
            if a == "act":
                break
            acq[i] += 1
            if a == "v2":
                x = int(q.v2[k])
                s, k = s + x, k + 1
                cost[i] += cm.per_query["sample"][i]
            elif a == "v3":
                o3 = q.v3
                cost[i] += cm.per_query["v3"][i]
            else:
                o4 = q.v4
                cost[i] += cm.per_query["v4"][i]
        lik = model.v2.lik(s, k, g)
        if o3 is not None:
            lik = lik * T3[:, o3]
        if o4 is not None:
            lik = lik * T4[:, o4]
        b = normalize(b_gen * lik)
        if value_escalate(b) - lam * C["llm"] > value_accept(b):
            perf[i], cost[i] = sig.llm_correct[i], cost[i] + cm.per_query["llm"][i]
        else:
            perf[i] = sig.slm_correct[i]
    return Outcome(perf, cost, {"lam": lam, "skip_rate": n_skip / n, "mean_acquisitions": float(acq.mean())})


def lambda_grid(cm: CostModel, n: int = 24) -> np.ndarray:
    """lam values spanning 'always escalate' to 'never escalate'.

    lam * C_llm is the accuracy gain an LLM call must buy to be worth it; below
    ~0.002 everything escalates and above 1 nothing does, so the grid spans that range.
    """
    return np.geomspace(2e-3, 1.0, n) / cm.mean["llm"]
