"""Belief over latent query types and the observation model of each verifier.

A query's hidden type z is one of TYPES (easy, hard, unsolvable, slm_only).
    V0  query-only prior     P(z | x): multinomial logistic regression on the embedding
    g   query group          k-means cluster of the embedding (learned from query text only)
    V1  answer confidence    P(bin | z, g), mean token log-prob cut into quantile bins
    V2  self-consistency     beta-binomial agreement model per (z, g); samples arrive one by one
    V3  retrieval verifier   P(#Correct votes | z, g)
    V4  context-free verif.  P(#Correct votes | z, g)

Signals are combined naive-Bayes style (independent given z and g). Every
signal is thereby put on one scale: how far it moves the belief over z.
Group-level tables are shrunk toward the global table so sparse cells stay sane.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import betaln, gammaln
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression

from . import TYPES

NZ = len(TYPES)
E, H, U, S = (TYPES.index(t) for t in ("easy", "hard", "unsolvable", "slm_only"))


def value_accept(b: np.ndarray) -> float:
    """Expected accuracy of keeping the SLM answer."""
    return b[..., E] + b[..., S]


def value_escalate(b: np.ndarray) -> float:
    """Expected accuracy of the LLM answer."""
    return b[..., E] + b[..., H]


def normalize(b: np.ndarray) -> np.ndarray:
    return b / b.sum(axis=-1, keepdims=True)


class DiscreteObs:
    """P(level | z, g) with two-level Dirichlet smoothing: group -> global -> uniform."""

    def __init__(self, n_levels: int, a_global: float = 1.0, a_group: float = 10.0):
        self.L, self.a0, self.a1 = n_levels, a_global, a_group

    def fit(self, level, z, g, n_groups, w=None):
        cnt = np.zeros((n_groups, NZ, self.L))
        np.add.at(cnt, (g, z, level), 1.0 if w is None else w)
        glob = cnt.sum(0)
        glob = (glob + self.a0 / self.L) / (glob.sum(-1, keepdims=True) + self.a0)
        self.table = (cnt + self.a1 * glob) / (cnt.sum(-1, keepdims=True) + self.a1)  # (G, NZ, L)
        return self

    def lik(self, level: int, g: int) -> np.ndarray:
        return self.table[g, :, level]


class KDEObs:
    """p(x | z) for a continuous signal: Gaussian KDE per type (Scott's bandwidth),
    shrunk toward the pooled KDE so rare types stay sane. Proposal section 4.3, as in AutoMix."""

    def __init__(self, a_shrink: float = 10.0):
        self.a = a_shrink

    def fit(self, x, z):
        from scipy.stats import gaussian_kde

        self.glob = gaussian_kde(x)
        self.kdes, self.n = [], np.bincount(z, minlength=NZ)
        for t in range(NZ):
            xt = x[z == t]
            self.kdes.append(gaussian_kde(xt) if len(xt) >= 5 and np.std(xt) > 0 else None)
        return self

    def lik(self, x: float) -> np.ndarray:
        g = float(self.glob(x)[0])
        out = np.empty(NZ)
        for t in range(NZ):
            kt = float(self.kdes[t](x)[0]) if self.kdes[t] is not None else g
            out[t] = (self.n[t] * kt + self.a * g) / (self.n[t] + self.a)
        return np.maximum(out, 1e-12)


class Binner:
    """Quantile bins fitted on the training split."""

    def __init__(self, n_bins: int = 6):
        self.n_bins = n_bins

    def fit(self, x, w=None):
        qs = np.linspace(0, 1, self.n_bins + 1)[1:-1]
        if w is None or np.allclose(w, w[0]):
            self.edges = np.quantile(x, qs)
        else:  # weighted quantiles
            o = np.argsort(x)
            cw = np.cumsum(w[o]) / np.sum(w)
            self.edges = x[o][np.minimum(np.searchsorted(cw, qs), len(x) - 1)]
        return self

    def __call__(self, x):
        return np.searchsorted(self.edges, x, side="right")


def _bb_logpmf_seq(s, k, a, b):
    """Log-probability of one particular sequence with s agreements in k draws."""
    return betaln(a + s, b + k - s) - betaln(a, b)


def _log_comb(n, r):
    return gammaln(n + 1) - gammaln(r + 1) - gammaln(n - r + 1)


class BetaBinomialObs:
    """Agreement draws per query: theta ~ Beta(alpha_zg, beta_zg), each sample agrees w.p. theta.

    The mean is fitted per (z, g), shrunk toward the per-z mean; the concentration
    alpha + beta is fitted per z by maximum likelihood.
    """

    def __init__(self, a_group: float = 10.0):
        self.a1 = a_group

    def fit(self, agree, z, g, n_groups, w=None):
        K = agree.shape[1]
        s = agree.sum(1)
        w = np.ones(len(s)) if w is None else w
        self.K = K
        mean_z = np.zeros(NZ)
        phi_z = np.zeros(NZ)
        for t in range(NZ):
            st, wt = s[z == t], w[z == t]
            if wt.sum() < 3:  # too few to fit: weakly informative default
                mean_z[t], phi_z[t] = (np.average(st, weights=wt) / K if wt.sum() > 0 else 0.5), 2.0
                continue

            def nll(p):
                a, b = np.exp(p)
                return -np.sum(wt * (_log_comb(K, st) + betaln(a + st, b + K - st) - betaln(a, b)))

            m0 = np.clip(np.average(st, weights=wt) / K, 0.02, 0.98)
            res = minimize(nll, np.log([m0 * 4, (1 - m0) * 4]), method="Nelder-Mead")
            a, b = np.exp(res.x)
            mean_z[t], phi_z[t] = a / (a + b), min(a + b, 200.0)
        self.alpha = np.zeros((n_groups, NZ))
        self.beta = np.zeros((n_groups, NZ))
        for gg in range(n_groups):
            for t in range(NZ):
                sel = (g == gg) & (z == t)
                m = ((w[sel] * s[sel]).sum() + self.a1 * mean_z[t] * K) / (K * (w[sel].sum() + self.a1))
                m = np.clip(m, 1e-3, 1 - 1e-3)
                self.alpha[gg, t], self.beta[gg, t] = m * phi_z[t], (1 - m) * phi_z[t]
        self._lik, self._pred = {}, {}  # memo: routers ask for the same few (s, k) states constantly
        return self

    def lik(self, s: int, k: int, g: int) -> np.ndarray:
        """Likelihood (over z) of having seen s agreements in the first k samples."""
        key = (s, k, g)
        if key not in self._lik:
            self._lik[key] = np.exp(_bb_logpmf_seq(s, k, self.alpha[g], self.beta[g]))
        return self._lik[key]

    def predictive(self, s: int, k: int, m: int, g: int) -> np.ndarray:
        """(NZ, m+1): P(s' agreements among the next m samples | z, s of k seen)."""
        key = (s, k, m, g)
        if key not in self._pred:
            a = self.alpha[g][:, None] + s
            b = self.beta[g][:, None] + (k - s)
            sp = np.arange(m + 1)[None, :]
            self._pred[key] = np.exp(_log_comb(m, sp) + betaln(a + sp, b + m - sp) - betaln(a, b))
        return self._pred[key]


class BeliefModel:
    """Everything a router needs to turn observed signals into a belief over z."""

    def __init__(self, n_groups: int = 4, n_v1_bins: int = 6, prior_C: float = 1.0, seed: int = 0,
                 v1_model: str = "bins"):
        self.n_groups, self.n_v1_bins, self.prior_C, self.seed = n_groups, n_v1_bins, prior_C, seed
        # "bins": quantile bins (main results); "kde": continuous likelihood as in the proposal,
        # which gave indistinguishable results (see the ablation table)
        self.v1_model = v1_model

    def fit(self, sig, w=None):
        """w: optional per-row weights, for soft labels spread over duplicated rows
        (see signals.soft_labeled). The query groups are fitted on the distinct queries."""
        z = sig.z
        if self.n_groups > 1:
            first = np.sort(np.unique(sig.emb, axis=0, return_index=True)[1])  # distinct rows, original order
            uniq = sig.emb[first]
            self.km = KMeans(self.n_groups, n_init=10, random_state=self.seed).fit(uniq)
        else:
            self.km = None
        g = self.group(sig.emb)
        self.lr = LogisticRegression(C=self.prior_C, max_iter=2000).fit(sig.emb, z, sample_weight=w)
        self.freq = np.bincount(z, weights=w, minlength=NZ) / (len(z) if w is None else np.sum(w))
        # The binned table serves the router's lookahead (it needs a finite outcome set);
        # the belief update uses the KDE when v1_model == "kde".
        self.binner = Binner(self.n_v1_bins).fit(sig.v1, w)
        self.v1 = DiscreteObs(self.n_v1_bins).fit(self.binner(sig.v1), z, g, self.n_groups, w)
        self.v1_kde = None
        if self.v1_model == "kde" and np.std(sig.v1) > 0:
            try:
                self.v1_kde = KDEObs().fit(sig.v1, z)
            except np.linalg.LinAlgError:  # degenerate data: keep the binned model
                self.v1_kde = None
        self.v2 = BetaBinomialObs().fit(sig.v2, z, g, self.n_groups, w)
        self.v3 = DiscreteObs(sig.n_verify + 1).fit(sig.v3, z, g, self.n_groups, w)
        self.v4 = DiscreteObs(sig.n_verify + 1).fit(sig.v4, z, g, self.n_groups, w)
        self.K = sig.v2.shape[1]
        return self

    def v1_lik(self, x: float, g: int) -> np.ndarray:
        """Likelihood over z of observing answer confidence x."""
        if self.v1_kde is None:
            return self.v1.lik(int(self.binner(x)), g)
        cache = self.__dict__.setdefault("_v1_cache", {})
        if x not in cache:
            cache[x] = self.v1_kde.lik(x)
        return cache[x]

    def group(self, emb) -> np.ndarray:
        if self.km is None:
            return np.zeros(len(emb), dtype=int)
        return self.km.predict(emb)

    def prior(self, emb) -> np.ndarray:
        """V0: P(z | x), shape (n, NZ), lightly smoothed toward the base rates."""
        p = np.zeros((len(emb), NZ))
        p[:, self.lr.classes_] = self.lr.predict_proba(emb)
        return normalize(0.95 * p + 0.05 * self.freq + 1e-6)
