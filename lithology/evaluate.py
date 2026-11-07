"""Scoring against hand-labelled truth, and the standard check on the organisers' example.

    python -m lithology.evaluate          # runs the example check (about 3 s)

Metrics:
  row ARI       every pixel row gets the predicted cluster and the true lithology, then the
                Adjusted Rand Index compares them (1 = perfect, ~0 = random). It doesn't
                depend on how the column was cut, so raw and merged intervals compare fairly.
  boundaries    recall / precision of predicted cuts within BOUNDARY_TOL px of true ones.
"""

import numpy as np
from sklearn.metrics import adjusted_rand_score

from .data import EXAMPLE_ANNOTATION, EXAMPLE_IMAGE, EXAMPLE_TRUTH

BOUNDARY_TOL = 20   # px. example.csv snaps boundaries to the 10 ft depth grid, which sits up to
                    # ~13 px from the visible change, so a tighter tolerance scores the grid, not us.


def row_labels(intervals, key, y0, y1):
    """Label of every pixel row in [y0, y1) taken from intervals[key] (None where uncovered)."""
    out = np.full(y1 - y0, None, dtype=object)
    for r in intervals:
        a, b = max(r["top_y"], y0), min(r["bottom_y"], y1)
        if b > a:
            out[a - y0: b - y0] = r[key]
    return out


def score(truth, merged, raw_n):
    """Row ARI and boundary recall/precision of `merged` against `truth` (both lists of dicts)."""
    y0 = min(r["top_y"] for r in truth)
    y1 = max(r["bottom_y"] for r in truth)
    t = row_labels(truth, "lithology", y0, y1)
    p = row_labels(merged, "cluster_id", y0, y1)
    ok = np.array([a is not None and b is not None for a, b in zip(t, p)])
    ari = adjusted_rand_score(t[ok].astype(str), p[ok].astype(str))
    tb = [r["top_y"] for r in truth[1:]]
    pb = [r["top_y"] for r in merged[1:] if y0 < r["top_y"] < y1]
    hit_t = sum(any(abs(a - b) <= BOUNDARY_TOL for b in pb) for a in tb)
    hit_p = sum(any(abs(a - b) <= BOUNDARY_TOL for b in tb) for a in pb)
    return dict(row_ari=ari, n_true=len(truth), n_raw=raw_n, n_merged=len(merged),
                b_recall=hit_t / max(1, len(tb)), b_precision=hit_p / max(1, len(pb)))


def run_example(quiet=False):
    """Cluster the example with its true intervals and with our segmentation, for both texture
    options. Returns {(input, texture): result dict}."""
    from .cluster import run          # imported here: cluster imports this module's score()
    results = {}
    for source, iv in (("true intervals", EXAMPLE_TRUTH), ("segmentation", None)):
        for tex in ("pooled", "team"):
            tag = f"{'truth' if iv else 'segmented'}_{tex}"
            results[(source, tex)] = run(EXAMPLE_IMAGE, EXAMPLE_ANNOTATION, iv, EXAMPLE_TRUTH,
                                         texture=tex, tag=tag, quiet=True)
    if not quiet:
        print(f"{'input intervals':16s} {'texture':7s} {'clusters':>8s} {'row ARI':>8s} "
              f"{'intervals in -> out':>20s} {'b.recall':>8s} {'b.prec':>7s} {'thin':>5s}")
        for (source, tex), r in results.items():
            print(f"{source:16s} {tex:7s} {r['n_clusters']:8d} {r['row_ari']:8.3f} "
                  f"{r['n_raw']:>9d} -> {r['n_merged']:<8d} {r['b_recall']:8.2f} "
                  f"{r['b_precision']:7.2f} {r['n_thin']:5d}")
        n_true = next(iter(results.values()))["n_true"]
        print(f"(truth: {n_true} intervals, 3 lithologies)  outputs in out/")
    return results


if __name__ == "__main__":
    run_example()
