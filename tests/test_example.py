"""Regression check on the organisers' example (the only document with ground truth).

    pytest tests/          (from the repo root; ~5 s)

If a change makes these fail, it made the example worse. The thresholds sit just below
the current scores so small, harmless changes don't trip them.
"""

from lithology.evaluate import run_example


def test_example_scores():
    r = run_example(quiet=True)

    truth = r[("true intervals", "pooled")]
    assert truth["n_clusters"] == 3            # three rocks: shale, siltstone, sandstone
    assert truth["row_ari"] > 0.99

    seg = r[("segmentation", "pooled")]
    assert seg["b_recall"] >= 0.95             # finds (almost) every true boundary
    assert seg["b_precision"] >= 0.90
    assert seg["row_ari"] > 0.95
