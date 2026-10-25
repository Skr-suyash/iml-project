"""Synthetic cache with the same schema as the real one, for testing the router side.

The generative story is built so the method has something to find: self-consistency
(V2) is informative on math, retrieval verification (V3) on factual questions, and
context-free self-verification (V4) is weak everywhere. Numbers here are made up;
results on synthetic data say nothing about real models.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .signals import Signals, from_records


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def make_records(n: int = 2800, d: int = 32, k_samples: int = 8, n_verify: int = 4, seed: int = 0):
    rng = np.random.default_rng(seed)
    task = np.where(rng.random(n) < 0.47, "math", "factual")
    centers = {t: rng.normal(size=d) for t in ("math", "factual")}
    w = rng.normal(size=d)
    diff = rng.normal(size=n)
    emb = np.stack([centers[t] for t in task]) + 0.6 * diff[:, None] * w / np.linalg.norm(w) * 3 \
        + 0.5 * rng.normal(size=(n, d))
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)

    is_math = task == "math"
    p_s = _sigmoid(np.where(is_math, 1.0, -0.3) - 1.5 * diff)
    p_l = _sigmoid(np.where(is_math, 2.5, 1.0) - 1.0 * diff)
    u = rng.random(n)
    slm = (u < p_s).astype(int)
    llm = (u < p_l).astype(int)
    flip = rng.random(n) < 0.05  # some independence between the two models
    llm[flip] = (rng.random(flip.sum()) < p_l[flip]).astype(int)

    v1 = rng.normal(np.where(is_math, -0.35 + 0.15 * slm, -0.6 + 0.25 * slm), 0.12)
    a = np.select([is_math & (slm == 1), is_math & (slm == 0), ~is_math & (slm == 1)], [8.0, 1.5, 5.0], 2.0)
    b = np.select([is_math & (slm == 1), is_math & (slm == 0), ~is_math & (slm == 1)], [1.5, 3.0, 2.0], 2.5)
    theta = rng.beta(a, b)
    v2 = (rng.random((n, k_samples)) < theta[:, None]).astype(int)
    p3 = np.where(is_math, np.where(slm == 1, 0.6, 0.5), np.where(slm == 1, 0.85, 0.25))
    p4 = np.where(slm == 1, 0.7, 0.5)

    rows = []
    for i in range(n):
        out = 200 if is_math[i] else 15
        rows.append({
            "qid": f"syn-{i:05d}", "task": str(task[i]), "question": "", "slm_answer": "", "llm_answer": "",
            "slm_correct": int(slm[i]), "llm_correct": int(llm[i]),
            "v1": {"mean_lp": float(v1[i]), "min_lp": float(v1[i] * 4), "perplexity": float(np.exp(-v1[i]))},
            "v2": v2[i].tolist(),
            "v3": [int(x) for x in rng.random(n_verify) < p3[i]],
            "v4": [int(x) for x in rng.random(n_verify) < p4[i]],
            "tokens": {"slm_in": 80, "slm_out": out, "sample_out": out, "v3_in": 600, "v3_out": 4 * 60,
                       "v4_in": 120 + out, "v4_out": 4 * 60, "llm_in": 80, "llm_out": out},
        })
    return rows, emb.astype("float32")


def make_signals(**kw) -> Signals:
    rows, emb = make_records(**kw)
    return from_records(rows, emb)


def write_cache(out_dir, **kw) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows, emb = make_records(**kw)
    with open(out / "cache.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    np.save(out / "embeddings.npy", emb)
    (out / "embeddings_qids.json").write_text(json.dumps([r["qid"] for r in rows]))
    return out
