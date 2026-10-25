"""Cost-aware verifier selection for context-free LLM cascades.

Generation side (runs on a GPU box: Kaggle / Colab):
    data, prompts, backends, retrieval, generate
Router side (runs anywhere, CPU only, from the cached outputs):
    grading, signals, costs, belief, routers, evaluate, simulate
"""

TYPES = ("easy", "hard", "unsolvable", "slm_only")
"""Latent query types, keyed by (SLM correct, LLM correct):
easy = (1, 1), hard = (0, 1), unsolvable = (0, 0), slm_only = (1, 0)."""
