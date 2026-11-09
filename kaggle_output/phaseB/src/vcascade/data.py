"""Load benchmark datasets into one shuffled query stream.

Every query is a plain dict so it serialises straight to JSONL:
    {"qid": "gsm8k-0012", "task": "math", "question": "...", "gold": {...}}
The `task` field is only used for grading, analysis and the oracle
per-category baseline. Routers never see it.
"""

from __future__ import annotations

import random


def load_gsm8k(limit: int | None = None) -> list[dict]:
    from datasets import load_dataset

    rows = load_dataset("openai/gsm8k", "main", split="test")
    out = []
    for i, r in enumerate(rows):
        final = r["answer"].split("####")[-1].strip().replace(",", "")
        out.append({
            "qid": f"gsm8k-{i:04d}",
            "task": "math",
            "question": r["question"].strip(),
            "gold": {"answer": final},
        })
    return out[:limit] if limit else out


def load_triviaqa(limit: int = 1500, seed: int = 0) -> list[dict]:
    from datasets import load_dataset

    rows = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="validation")
    first = {}
    for i, qid in enumerate(rows["question_id"]):  # the split repeats some questions
        first.setdefault(qid, i)
    idx = sorted(first.values())
    random.Random(seed).shuffle(idx)
    out = []
    for i in idx[:limit]:
        r = rows[i]
        out.append({
            "qid": f"tqa-{r['question_id']}",
            "task": "factual",
            "question": r["question"].strip(),
            "gold": {
                "answer": r["answer"]["value"],
                "aliases": sorted(set(r["answer"]["normalized_aliases"])),
            },
        })
    return out


LOADERS = {"math": load_gsm8k, "factual": load_triviaqa}


def build_stream(tasks=("math", "factual"), limits: dict | None = None, seed: int = 0) -> list[dict]:
    """Concatenate the requested tasks and shuffle them into one stream."""
    limits = limits or {}
    queries = []
    for t in tasks:
        queries += LOADERS[t](limit=limits.get(t)) if limits.get(t) else LOADERS[t]()
    random.Random(seed).shuffle(queries)
    return queries
