import numpy as np
import pytest

from vcascade import evaluate, routers
from vcascade.belief import BeliefModel, normalize
from vcascade.costs import CostModel
from vcascade.simulate import make_signals


@pytest.fixture(scope="module")
def setup():
    sig = make_signals(n=1200, seed=1)
    idx = np.random.default_rng(0).permutation(len(sig))
    tr, te = sig.subset(np.sort(idx[:400])), sig.subset(np.sort(idx[400:]))
    model = BeliefModel(n_groups=3).fit(tr)
    cm = CostModel.unit(len(te), ratio=10)
    return tr, te, model, cm


def test_observation_tables_are_distributions(setup):
    _, _, m, _ = setup
    for obs in (m.v1, m.v3, m.v4):
        assert np.allclose(obs.table.sum(-1), 1)
    pred = m.v2.predictive(s=2, k=3, m=4, g=0)
    assert np.allclose(pred.sum(1), 1)


def test_beta_binomial_sequential_matches_batch(setup):
    """Updating one sample at a time must equal the batch likelihood of the sequence."""
    _, _, m, _ = setup
    seq = [1, 0, 1, 1]
    b_seq = np.full(4, 0.25)
    s = 0
    for k, a in enumerate(seq):
        b_seq = normalize(b_seq * m.v2.predictive(s, k, 1, 0)[:, a])
        s += a
    b_batch = normalize(np.full(4, 0.25) * m.v2.lik(sum(seq), len(seq), 0))
    assert np.allclose(b_seq, b_batch)


def test_greedy_extremes(setup):
    _, te, m, cm = setup
    # lam = 0: cost is free, so the router never settles for less than the best decision
    free = routers.greedy_voi(m, te, cm, lam=0.0)
    # huge lam: nothing is worth buying, nor escalating
    stingy = routers.greedy_voi(m, te, cm, lam=10.0)
    assert stingy.cost.mean() <= cm.mean["gen"] + 1e-9
    assert free.perf.mean() >= stingy.perf.mean()


def test_cost_accuracy_monotone_in_lambda(setup):
    _, te, m, cm = setup
    pts = [routers.greedy_voi(m, te, cm, lam).point for lam in routers.lambda_grid(cm, 8)]
    costs = [c for c, _ in pts]
    assert costs[0] >= costs[-1]


def test_exact_at_least_as_good_as_greedy_in_objective(setup):
    _, te, m, cm = setup
    sub = np.arange(60)
    te_s, cm_s = te.subset(sub), cm.subset(sub)
    lam = 0.02 / cm.mean["llm"]
    g = routers.greedy_voi(m, te_s, cm_s, lam, v2_batches=(1,))
    e = routers.exact_planner(m, te_s, cm_s, lam, max_k=8)
    # Expected objective is optimal for the planner; realized outcomes are noisy, so allow slack.
    obj = lambda o: o.perf.mean() - lam * o.cost.mean()
    assert obj(e) >= obj(g) - 0.05


def test_metrics_on_random_line_are_zero():
    slm, llm = (1.0, 0.5), (11.0, 0.8)
    pts = [(1 + 10 * t, 0.5 + 0.3 * t) for t in np.linspace(0, 1, 50)]
    fr = evaluate.frontier(pts)
    assert abs(evaluate.ibc_lift(fr, slm, llm)) < 5
    assert abs(evaluate.area_lift(fr, slm, llm)) < 0.5
