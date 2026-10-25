"""Run every generation stage with a fake backend, then load the cache on the router side."""

import json

import numpy as np

from vcascade import generate, retrieval, signals
from vcascade.costs import CostModel
from vcascade import routers
from vcascade.belief import BeliefModel

QUERIES = [
    {"qid": f"m{i}", "task": "math", "question": f"What is {i} + {i}?", "gold": {"answer": str(2 * i)}}
    for i in range(12)
] + [
    {"qid": f"f{i}", "task": "factual", "question": f"Capital of country {i}?",
     "gold": {"answer": f"City {i}", "aliases": [f"city {i}"]}}
    for i in range(12)
]


class FakeBackend:
    """Answers math right for even i, factual right for i < 6; verifies by keyword."""

    def __init__(self, strong: bool):
        self.strong = strong

    def generate(self, conversations, n=1, temperature=0.0, max_tokens=512, logprobs=False, seed=0):
        out = []
        for conv in conversations:
            text = conv[-1]["content"]
            if "Verdict" in text:
                reply = "Looks right. Verdict: Correct" if "City 1" in text or "4" in text else "Verdict: Incorrect"
            elif "+" in text:
                i = int(text.split("What is ")[1].split(" +")[0])
                ok = self.strong or i % 2 == 0
                reply = f"{i} + {i} = {2 * i if ok else 2 * i + 1}\nFinal answer: {2 * i if ok else 2 * i + 1}"
            else:
                i = int(text.split("country ")[1].split("?")[0])
                reply = f"Final answer: City {i if (self.strong or i < 6) else 99}"
            comps = [{"text": reply, "token_logprobs": [-0.1, -0.5] if logprobs else None,
                      "prompt_tokens": 50, "completion_tokens": 10} for _ in range(n)]
            out.append(comps)
        return out


def test_pipeline_end_to_end(tmp_path, monkeypatch):
    generate.write_jsonl(tmp_path / "queries.jsonl", QUERIES)
    monkeypatch.setattr(generate, "_backend", lambda args: FakeBackend(strong="7B" in args.model))
    monkeypatch.setattr(retrieval, "retrieve_all",
                        lambda qs, k=3, **kw: [[{"title": "T", "text": "some evidence"}] for _ in qs])
    for stage, extra in [("slm", ["--model", "slm-1.5B"]), ("retrieve", []),
                         ("verify", ["--model", "slm-1.5B"]), ("llm", ["--model", "llm-7B"]), ("merge", [])]:
        generate.main(["--out", str(tmp_path), stage, *extra])

    rows = generate.read_jsonl(tmp_path / "cache.jsonl")
    by_id = {r["qid"]: r for r in rows}
    assert by_id["m2"]["slm_correct"] == 1 and by_id["m3"]["slm_correct"] == 0
    assert by_id["f3"]["slm_correct"] == 1 and by_id["f9"]["slm_correct"] == 0
    assert all(r["llm_correct"] == 1 for r in rows)
    assert len(by_id["m0"]["v2"]) == 8 and len(by_id["m0"]["v3"]) == 4

    # router side reads it
    np.save(tmp_path / "embeddings.npy", np.random.default_rng(0).normal(size=(len(rows), 8)).astype("float32"))
    (tmp_path / "embeddings_qids.json").write_text(json.dumps([r["qid"] for r in rows]))
    sig = signals.load(tmp_path)
    model = BeliefModel(n_groups=2).fit(sig)
    out = routers.greedy_voi(model, sig, CostModel.unit(len(sig), 10), lam=0.01)
    assert out.perf.shape == (len(sig),)
