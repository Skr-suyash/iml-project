"""Probe signal: a linear classifier on the SLM's hidden states that predicts its correctness.

The probe replaces V1 (mean token log-prob) as the free post-generation signal: it is
computed from the same forward pass that produced the answer, so it costs nothing extra.
V1 itself is one of the probe's input features, so the probe can only add to it.

Everything is fitted on the training split. The training split's own probe scores are
out-of-fold (5-fold), so the belief model's likelihood table for the probe is fitted on
scores as noisy as the ones it will see at test time.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .signals import Signals

POOLS = ("last", "mean")
CS = (1e-3, 1e-2, 1e-1)


def features(sig: Signals, pool: str, layer: int) -> np.ndarray:
    return np.column_stack([sig.hidden[pool][:, layer].astype(np.float32), sig.v1])


def _clf(C: float):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000))


class Probe:
    """Chooses pooling, layer and regularisation by cross-validated AUROC on the training split."""

    def fit(self, sig: Signals, y: np.ndarray | None = None, seed: int = 0):
        y = sig.slm_correct if y is None else y
        cv = StratifiedKFold(5, shuffle=True, random_state=seed)
        best = (-1.0, None)
        for pool in POOLS:
            for layer in range(sig.hidden[pool].shape[1]):
                x = features(sig, pool, layer)
                for C in CS:
                    oof = cross_val_predict(_clf(C), x, y, cv=cv, method="predict_proba")[:, 1]
                    auc = roc_auc_score(y, oof)
                    if auc > best[0]:
                        best = (auc, (pool, layer, C, oof))
        self.cv_auroc = best[0]
        self.pool, self.layer, self.C, self.oof = best[1]
        self.clf = _clf(self.C).fit(features(sig, self.pool, self.layer), y)
        return self

    def score(self, sig: Signals) -> np.ndarray:
        return self.clf.predict_proba(features(sig, self.pool, self.layer))[:, 1]

    @property
    def config(self) -> dict:
        return {"pool": self.pool, "layer": int(self.layer), "C": self.C, "cv_auroc": float(self.cv_auroc)}


def with_probe(tr: Signals, te: Signals, y_tr: np.ndarray | None = None, seed: int = 0):
    """(train, test) signals with V1 replaced by the probe's P(SLM correct).

    y_tr: probe training target (default: the train split's SLM correctness; pass LLM
    agreement for label-free training).
    """
    p = Probe().fit(tr, y_tr, seed)
    return replace(tr, v1=p.oof), replace(te, v1=p.score(te)), p


def slm_prior(tr: Signals, te: Signals, layer: int):
    """(train, test) signals whose query representation (V0 prior and query groups) is the
    SLM's own hidden state at the end of the prompt, instead of a sentence embedding.

    Available after the prompt's prefill, before any answer token, so the skip decision can use it.
    """
    x_tr, x_te = (s.hidden["prompt"][:, layer].astype(np.float32) for s in (tr, te))
    sc = StandardScaler().fit(x_tr)
    d = x_tr.shape[1]
    return replace(tr, emb=sc.transform(x_tr) / np.sqrt(d)), replace(te, emb=sc.transform(x_te) / np.sqrt(d))
