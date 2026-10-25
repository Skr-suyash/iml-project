"""Cost-accuracy curves and summary metrics.

A method's curve is the set of (mean cost, mean accuracy) points it reaches
across its threshold or lambda sweep. We summarise it by its frontier: the upper
convex hull of those points (randomising between two operating points reaches any
point on the segment between them). The always-SLM and always-LLM points are added
to every frontier, since every user has those options, so no method scores below
the random-mixing line.

IBC lift (AutoMix, Aggarwal et al. 2024, Eq. 1):
    IBC_M    = (P_M - P_SLM) / (C_M - C_SLM)
    IBC_base = (P_LLM - P_SLM) / (C_LLM - C_SLM)
    lift     = (IBC_M - IBC_base) / IBC_base * 100
evaluated at the midpoints of five equal cost regions between C_SLM and C_LLM
and averaged, mirroring the paper's five-region average.

Area lift: mean accuracy gain over the random-mixing line between C_SLM and
C_LLM, in accuracy points (x100).
"""

from __future__ import annotations

import numpy as np


def frontier(points) -> np.ndarray:
    """Upper convex hull of (cost, perf) points, cost ascending, perf strictly increasing.

    Randomising between any two operating points reaches every point on the segment
    joining them, so the hull, not the raw curve, is what a method can deliver.
    """
    pts = sorted(set((float(c), float(p)) for c, p in points))
    hull = []
    for pt in pts:  # monotone chain, upper half
        while len(hull) >= 2:
            (x1, y1), (x2, y2) = hull[-2], hull[-1]
            if (x2 - x1) * (pt[1] - y1) - (y2 - y1) * (pt[0] - x1) >= 0:  # hull[-1] not above the chord
                hull.pop()
            else:
                break
        hull.append(pt)
    keep, best = [], -np.inf
    for c, p in hull:
        if p > best + 1e-12:
            keep.append((c, p))
            best = p
    return np.array(keep)


def perf_at(front: np.ndarray, cost, p_slm: float) -> np.ndarray:
    """Accuracy on the frontier at `cost`: linear between points, flat beyond the last."""
    cost = np.atleast_1d(cost)
    out = np.interp(cost, front[:, 0], front[:, 1], left=p_slm, right=front[-1, 1])
    return np.maximum(out, p_slm)


def ibc_lift(front, slm, llm, n_regions: int = 5) -> float:
    (c_s, p_s), (c_l, p_l) = slm, llm
    base = (p_l - p_s) / (c_l - c_s)
    mids = c_s + (np.arange(n_regions) + 0.5) * (c_l - c_s) / n_regions
    ibc = (perf_at(front, mids, p_s) - p_s) / (mids - c_s)
    return float(np.mean((ibc - base) / base * 100))


def area_lift(front, slm, llm, n_grid: int = 400) -> float:
    (c_s, p_s), (c_l, p_l) = slm, llm
    grid = np.linspace(c_s, c_l, n_grid)
    rand = p_s + (grid - c_s) * (p_l - p_s) / (c_l - c_s)
    return float(np.mean(perf_at(front, grid, p_s) - rand) * 100)


def summarize(outcomes, slm, llm) -> dict:
    pts = [o.point for o in outcomes]
    fr = frontier(pts + [slm, llm])  # every user can also pick always-SLM or always-LLM
    return {"points": pts, "frontier": fr.tolist(),
            "ibc_lift": ibc_lift(fr, slm, llm), "area_lift": area_lift(fr, slm, llm),
            "max_perf": float(fr[:, 1].max())}


def paired_bootstrap(per_query_a, per_query_b, n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    """Mean difference a - b over queries with a 95% bootstrap interval half-width."""
    rng = np.random.default_rng(seed)
    d = np.asarray(per_query_a) - np.asarray(per_query_b)
    boots = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(d.mean()), float((hi - lo) / 2)
