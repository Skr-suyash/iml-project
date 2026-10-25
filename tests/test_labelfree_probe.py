from dataclasses import replace

import numpy as np
import pytest

from vcascade.probe import with_probe
from vcascade.signals import _loose_match, pseudo_labeled
from vcascade.simulate import make_signals


@pytest.mark.parametrize("a,b,same", [
    ("So, Seth is 16 years old.", "16", True),
    ("$85", "Jayden needs $85 more to buy the camera.", True),
    ("eight", "8", True),
    ("clogs", "clog", True),
    ("Benedict Cumberbatch", "Benedict Cumberbatch", True),
    ("16", "17", False),
    ("Andrew Johnson", "Andrew Jackson", False),
    (None, "16", False),
])
def test_loose_match(a, b, same):
    assert _loose_match(a, b) is same


def test_pseudo_labels_follow_llm_agreement():
    sig = make_signals(n=6, seed=0)
    sig = replace(sig, slm_answer=np.array(["1", "2", "x", "y", "5", None], dtype=object),
                  llm_answer=np.array(["1", "3", "x", "z", "5", "6"], dtype=object))
    p = pseudo_labeled(sig)
    assert p.slm_correct.tolist() == [1, 0, 1, 0, 1, 0]
    assert p.llm_correct.tolist() == [1] * 6
    keep = np.array([False, True, False, False, False, False])
    h = pseudo_labeled(sig, keep_true=keep)
    assert h.slm_correct[1] == sig.slm_correct[1] and h.llm_correct[1] == sig.llm_correct[1]
    assert h.slm_correct[0] == 1


def test_probe_recovers_a_planted_signal():
    sig = make_signals(n=600, seed=2)
    rng = np.random.default_rng(0)
    d = 16
    h = rng.normal(size=(len(sig), 2, d)).astype(np.float16)
    h[:, 1, 0] += 2.0 * sig.slm_correct  # layer 2, dim 0 carries correctness
    sig = replace(sig, hidden={"last": h, "mean": h.copy(), "prompt": h.copy()})
    tr, te = sig.subset(np.arange(300)), sig.subset(np.arange(300, 600))
    tr2, te2, p = with_probe(tr, te)
    from sklearn.metrics import roc_auc_score
    assert p.layer == 1
    assert len(tr2.v1) == len(tr) and len(te2.v1) == len(te)
    assert roc_auc_score(te.slm_correct, te2.v1) > roc_auc_score(te.slm_correct, te.v1)
    assert roc_auc_score(te.slm_correct, te2.v1) > 0.8


def test_unit_weights_match_unweighted_fit():
    from vcascade.belief import BeliefModel

    sig = make_signals(n=400, seed=3)
    a = BeliefModel(3).fit(sig)
    b = BeliefModel(3).fit(sig, np.ones(len(sig)))
    assert np.allclose(a.v1.table, b.v1.table) and np.allclose(a.v3.table, b.v3.table)
    assert np.allclose(a.v2.alpha, b.v2.alpha, rtol=1e-4) and np.allclose(a.freq, b.freq)
    assert np.allclose(a.prior(sig.emb[:20]), b.prior(sig.emb[:20]), atol=1e-6)


def test_soft_labels_spread_each_query_with_unit_weight():
    from vcascade.belief import BeliefModel
    from vcascade.signals import soft_labeled

    sig = make_signals(n=300, seed=4)
    ans = np.array([str(i % 7) for i in range(len(sig))], dtype=object)
    llm = np.where(sig.slm_correct == 1, ans, "other")
    sig = replace(sig, slm_answer=ans, llm_answer=llm.astype(object))
    gold = np.arange(len(sig)) < 50
    soft, w = soft_labeled(sig, gold, n_groups=3)
    per_query = {}
    for q, wi in zip(soft.qid, w):
        per_query[q] = per_query.get(q, 0) + wi
    assert np.allclose(list(per_query.values()), 1.0, atol=1e-2)  # tiny weights are dropped
    assert len(per_query) == len(sig)
    BeliefModel(3).fit(soft, w)  # fits without error
