"""Split-paired comparisons for the labelfree / probe results: mean area-lift difference and
how many of the splits the first method wins.

    python scripts/paired_splits.py results/ablations/probe_unit.json "Ours + probe" "Fixed V1 (log-prob threshold)"
"""

import json
import sys

import numpy as np


def compare(path, a, b, metric="area_lift"):
    res = json.loads(open(path, encoding="utf-8").read())["results"]
    out = {}
    for r, row in res.items():
        d = np.array([x[metric] for x in row[a]]) - np.array([x[metric] for x in row[b]])
        out[r] = (float(d.mean()), int((d > 0).sum()), len(d))
    return out


if __name__ == "__main__":
    path, a, b = sys.argv[1:4]
    for r, (m, wins, n) in compare(path, a, b).items():
        print(f"ratio {r}: {a} - {b} = {m:+.2f} area-lift points, wins {wins}/{n} splits")
