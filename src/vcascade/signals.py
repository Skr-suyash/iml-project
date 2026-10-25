"""Load cache.jsonl (+ embeddings) into numpy arrays the routers work on."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from . import TYPES


def type_index(slm_correct: np.ndarray, llm_correct: np.ndarray) -> np.ndarray:
    """easy = 0 (1,1), hard = 1 (0,1), unsolvable = 2 (0,0), slm_only = 3 (1,0)."""
    z = np.empty(len(slm_correct), dtype=int)
    z[(slm_correct == 1) & (llm_correct == 1)] = TYPES.index("easy")
    z[(slm_correct == 0) & (llm_correct == 1)] = TYPES.index("hard")
    z[(slm_correct == 0) & (llm_correct == 0)] = TYPES.index("unsolvable")
    z[(slm_correct == 1) & (llm_correct == 0)] = TYPES.index("slm_only")
    return z


def _votes(v) -> int:
    """Number of 'Correct' verdicts; unparseable verdicts count as not correct."""
    return sum(1 for x in v if x == 1)


@dataclass
class Signals:
    qid: np.ndarray            # (n,) str
    task: np.ndarray           # (n,) str; analysis and the oracle baseline only
    slm_correct: np.ndarray    # (n,) int
    llm_correct: np.ndarray    # (n,) int
    v1: np.ndarray             # (n,) mean token log-prob of the greedy SLM answer
    v2: np.ndarray             # (n, K) 1 if sample k agrees with the greedy answer
    v3: np.ndarray             # (n,) number of 'Correct' votes, retrieval-grounded verifier
    v4: np.ndarray             # (n,) number of 'Correct' votes, context-free verifier
    n_verify: int              # votes per verifier call
    emb: np.ndarray            # (n, d) query embedding for V0
    tokens: dict               # name -> (n,) token counts, for the token cost model
    slm_answer: np.ndarray | None = None  # (n,) extracted answers, for label-free training
    llm_answer: np.ndarray | None = None
    hidden: dict | None = None            # pool -> (n, layers, d) SLM hidden states, for the probe

    @property
    def z(self) -> np.ndarray:
        return type_index(self.slm_correct, self.llm_correct)

    def __len__(self) -> int:
        return len(self.qid)

    def subset(self, idx: np.ndarray) -> "Signals":
        def take(x):
            return None if x is None else x[idx]

        return Signals(qid=self.qid[idx], task=self.task[idx], slm_correct=self.slm_correct[idx],
                       llm_correct=self.llm_correct[idx], v1=self.v1[idx], v2=self.v2[idx], v3=self.v3[idx],
                       v4=self.v4[idx], n_verify=self.n_verify, emb=self.emb[idx],
                       tokens={k: v[idx] for k, v in self.tokens.items()},
                       slm_answer=take(self.slm_answer), llm_answer=take(self.llm_answer),
                       hidden=None if self.hidden is None else {k: v[idx] for k, v in self.hidden.items()})


_NUM_WORDS = {w: str(i) for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve".split())}


def _loose_match(a, b) -> bool:
    """answers_match, plus: a bare number found inside a sentence ("Seth is 16 years old" vs "16"),
    number words ("eight" vs "8") and singular/plural ("clogs" vs "clog")."""
    from .grading import _NUM, answers_match, canon_number, canonical

    if answers_match(a, b):
        return True
    ka, kb = canonical(a), canonical(b)
    if ka is None or kb is None:
        return False
    ka, kb = (" ".join(_NUM_WORDS.get(w, w) for w in k.split()) for k in (ka, kb))
    for x, y in ((ka, kb), (kb, ka)):
        n = canon_number(x)
        if n is not None and n in {canon_number(m) for m in _NUM.findall(y.replace(",", ""))}:
            return True
    return ka.rstrip("s") == kb.rstrip("s")


def llm_agreement(sig: Signals) -> np.ndarray:
    """1 where the SLM's answer matches the LLM's answer.

    Looser than the self-consistency matcher: this only builds training labels, and the
    SLM often wraps its number in a sentence while the LLM answers with the bare number.
    """
    return np.array([int(_loose_match(a, b)) for a, b in zip(sig.slm_answer, sig.llm_answer)])


def pseudo_labeled(sig: Signals, keep_true: np.ndarray | None = None) -> Signals:
    """Training labels without a human: treat the LLM's answer as the truth.

    SLM correct := its answer matches the LLM's; LLM correct := 1. The latent types then
    collapse to easy (agree) and hard (disagree), and the router learns to predict
    disagreement. Needs both models run once on the training queries, but no gold answers.
    keep_true (bool mask or indices): queries whose real labels are kept (hybrid training).
    """
    slm = llm_agreement(sig)
    llm = np.ones(len(sig), dtype=int)
    if keep_true is not None:
        slm[keep_true], llm[keep_true] = sig.slm_correct[keep_true], sig.llm_correct[keep_true]
    return replace(sig, slm_correct=slm, llm_correct=llm)


_LABELS = {TYPES.index("easy"): (1, 1), TYPES.index("hard"): (0, 1),
           TYPES.index("unsolvable"): (0, 0), TYPES.index("slm_only"): (1, 0)}


def soft_labeled(sig: Signals, gold: np.ndarray, n_groups: int = 4, a_shrink: float = 5.0, seed: int = 0):
    """Calibrated stand-in labels from LLM agreement plus a few gold labels.

    LLM agreement tells whether the SLM is probably right, but not whether the LLM can fix
    a disagreement or both models are lost (unsolvable). A few gold labels estimate
    P(type | agreement, query group); every query without a gold label is then spread over
    the types with that probability (duplicated rows carrying weights). Gold-labelled
    queries keep their true type with weight 1. Group estimates are shrunk toward the
    pooled estimate with strength a_shrink.

    Returns (signals with duplicated rows, row weights) for BeliefModel.fit(sig, w).
    """
    from sklearn.cluster import KMeans

    gold = np.zeros(len(sig), bool) | np.isin(np.arange(len(sig)), np.flatnonzero(gold))
    agree = llm_agreement(sig)
    g = (KMeans(n_groups, n_init=10, random_state=seed).fit_predict(sig.emb) if n_groups > 1
         else np.zeros(len(sig), dtype=int))
    z = sig.z
    cnt = np.zeros((n_groups, 2, len(TYPES)))
    np.add.at(cnt, (g[gold], agree[gold], z[gold]), 1)
    pooled = cnt.sum(0) + 0.5
    pooled /= pooled.sum(-1, keepdims=True)
    p = (cnt + a_shrink * pooled) / (cnt.sum(-1, keepdims=True) + a_shrink)  # (G, 2, NZ)

    rows, slm, llm, w = [], [], [], []
    for i in range(len(sig)):
        if gold[i]:
            rows.append(i), slm.append(sig.slm_correct[i]), llm.append(sig.llm_correct[i]), w.append(1.0)
            continue
        for t, pt in enumerate(p[g[i], agree[i]]):
            if pt > 1e-3:
                rows.append(i), slm.append(_LABELS[t][0]), llm.append(_LABELS[t][1]), w.append(pt)
    out = replace(sig.subset(np.array(rows)), slm_correct=np.array(slm), llm_correct=np.array(llm))
    return out, np.array(w)


def from_records(rows: list[dict], emb: np.ndarray) -> Signals:
    n_verify = len(rows[0]["v3"])
    return Signals(
        qid=np.array([r["qid"] for r in rows]),
        task=np.array([r["task"] for r in rows]),
        slm_correct=np.array([r["slm_correct"] for r in rows], dtype=int),
        llm_correct=np.array([r["llm_correct"] for r in rows], dtype=int),
        v1=np.array([r["v1"]["mean_lp"] for r in rows], dtype=float),
        v2=np.array([r["v2"] for r in rows], dtype=int),
        v3=np.array([_votes(r["v3"]) for r in rows], dtype=int),
        v4=np.array([_votes(r["v4"]) for r in rows], dtype=int),
        n_verify=n_verify,
        emb=emb.astype(float),
        tokens={k: np.array([r["tokens"][k] for r in rows], dtype=float) for k in rows[0]["tokens"]},
        slm_answer=np.array([r.get("slm_answer") for r in rows], dtype=object),
        llm_answer=np.array([r.get("llm_answer") for r in rows], dtype=object),
    )


def load(cache_dir: str | Path) -> Signals:
    cache_dir = Path(cache_dir)
    with open(cache_dir / "cache.jsonl", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    emb = np.load(cache_dir / "embeddings.npy")
    order = json.loads((cache_dir / "embeddings_qids.json").read_text())
    pos = {q: i for i, q in enumerate(order)}
    emb = emb[[pos[r["qid"]] for r in rows]]
    sig = from_records(rows, emb)
    if (cache_dir / "hidden.npz").exists():  # optional: SLM hidden states for the probe signal
        h = np.load(cache_dir / "hidden.npz")
        hpos = {q: i for i, q in enumerate(h["qids"])}
        take = [hpos[q] for q in sig.qid]
        sig.hidden = {k: h[k][take] for k in ("prompt", "last", "mean")}
    return sig
