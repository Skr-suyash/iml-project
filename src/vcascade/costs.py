"""Cost models. All costs are in units of one SLM answer call.

unit  : fixed per-call costs; the LLM costs `ratio` units (the paper's sweep).
token : per-query costs from cached token counts and per-token prices,
        rescaled so the mean SLM answer costs 1 unit.

Routers decide with the mean cost of each action (the true cost of a call is
unknown until it returns); accounting uses each query's actual cost.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

ACTIONS = ("gen", "sample", "v3", "v4", "llm")


@dataclass
class CostModel:
    per_query: dict  # action -> (n,) cost of that action on each query

    @property
    def mean(self) -> dict:
        return {a: float(np.mean(c)) for a, c in self.per_query.items()}

    @classmethod
    def unit(cls, n: int, ratio: float, sample: float = 1.0, v3: float = 1.5, v4: float = 1.0) -> "CostModel":
        ones = np.ones(n)
        return cls({"gen": ones, "sample": sample * ones, "v3": v3 * ones, "v4": v4 * ones, "llm": ratio * ones})

    @classmethod
    def token(cls, tokens: dict, slm_price=(1.0, 1.0), llm_price=(10.0, 10.0),
              retrieval: float = 0.05) -> "CostModel":
        """Prices are (input, output) per token in any currency; only ratios matter.

        `retrieval` is the cost of one Wikipedia search, in SLM-answer units.
        """
        si, so = slm_price
        li, lo = llm_price
        gen = si * tokens["slm_in"] + so * tokens["slm_out"]
        scale = float(np.mean(gen))
        return cls({
            "gen": gen / scale,
            "sample": (si * tokens["slm_in"] + so * tokens["sample_out"]) / scale,
            "v3": (si * tokens["v3_in"] + so * tokens["v3_out"]) / scale + retrieval,
            "v4": (si * tokens["v4_in"] + so * tokens["v4_out"]) / scale,
            "llm": (li * tokens["llm_in"] + lo * tokens["llm_out"]) / scale,
        })

    def subset(self, idx: np.ndarray) -> "CostModel":
        return CostModel({a: c[idx] for a, c in self.per_query.items()})


class CostPredictor:
    """Per-query cost estimates for the router, fitted on the training split.

    Before generation only the query is known: ridge regression of log cost on the
    embedding. After generation the SLM's output length is also known, and it predicts
    the length (hence cost) of extra samples, verification calls and the LLM's answer.
    Under the unit cost model every estimate is the constant per-call cost.
    """

    def fit(self, emb, slm_out_tokens, cm: CostModel):
        from sklearn.linear_model import Ridge

        self.pre, self.post = {}, {}
        x_post = np.column_stack([emb, np.log1p(slm_out_tokens)])
        for a, c in cm.per_query.items():
            y = np.log(np.maximum(c, 1e-9))
            self.pre[a] = Ridge(alpha=1.0).fit(emb, y)
            self.post[a] = Ridge(alpha=1.0).fit(x_post, y)
        return self

    def predict(self, emb, slm_out_tokens) -> tuple[dict, dict]:
        x_post = np.column_stack([emb, np.log1p(slm_out_tokens)])
        pre = {a: np.exp(m.predict(emb)) for a, m in self.pre.items()}
        post = {a: np.exp(m.predict(x_post)) for a, m in self.post.items()}
        return pre, post
