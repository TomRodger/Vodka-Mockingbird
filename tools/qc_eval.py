"""Evidence behind lithology/qc.py: do the match signals separate right from wrong names?

    python -m tools.qc_eval

For every pixel row of every labelled section (tools/labels), compares the pipeline's name with
the label and prints:
  1. what kind of disagreement each wrong row is (wording, boundary offset or real error)
  2. the match signals (shape / colour distance, runner-up margin) on right vs wrong names
  3. how many rows each candidate per-interval flag would mark, and how many of those are wrong,
     with thresholds set leave-one-section-out (95th / 5th percentile of correct rows of the
     OTHER sections), so no count is measured on data that set its threshold.
Result when written (7 sections): the signals overlap and every flagged row was correct, so
these flags were dropped from lithology/qc.py.
"""

import glob
from collections import Counter

import numpy as np

import tools.label_tool as T


def rows_of(path):
    """(truth name, predicted row) for every pixel row of one labelled section."""
    stem, _ = T.parse_label_name(path)
    stem, img, _ = T.resolve(stem)
    truth = T.merge_same(T.read_labels(path))
    top, bot = truth[0][0], truth[-1][1]
    pred = T.predict(img, stem, top, bot)
    out = []
    for y in range(top, bot):
        t = next(n for a, b, n in truth if a <= y < b)
        p = next((r for r in pred if r["top_y"] <= y < r["bottom_y"]), None)
        if p is not None and t.strip().lower() != "no_data" and p["lithology"] != "no_data":
            out.append((t, p))
    return out


same = lambda t, p: t.strip().lower() == p["lithology"].strip().lower()
named = lambda p: p.get("match") and p["match"]["name"] != "unknown"


def thresholds_from(rows):
    ok = [p["match"] for t, p in rows if named(p) and same(t, p)]
    pick = lambda k, q: float(np.percentile([m[k] for m in ok if m[k] is not None], q))
    return dict(shape=pick("shape", 95), colour=pick("colour", 95), margin=pick("margin", 5))


def candidate_flags(m, th):
    out = []
    if m["shape"] is not None and m["shape"] > th["shape"]:
        out.append("poor_shape")
    if m["colour"] is not None and m["colour"] > th["colour"]:
        out.append("poor_colour")
    if m["margin"] is not None and m["margin"] < th["margin"]:
        out.append("near_tie")
    if m["prior"]:
        out.append("prior")
    return out


def main():
    files = sorted(glob.glob("tools/labels/*.txt"))
    data = {f: rows_of(f) for f in files}
    allrows = [r for f in files for r in data[f]]
    right = [p["match"] for t, p in allrows if named(p) and same(t, p)]
    wrong = [(t, p) for t, p in allrows if named(p) and not same(t, p)]
    print(f"rows named: {len(right) + len(wrong)}, right {len(right)}, wrong {len(wrong)}")
    print("wrong rows by (label, predicted):", Counter((t, p["lithology"]) for t, p in wrong).most_common(8))
    print("\nsignal       right p50 / p95      wrong p50 / max")
    for k in ("shape", "colour", "margin"):
        r = np.array([m[k] for m in right if m[k] is not None])
        w = np.array([p["match"][k] for _, p in wrong if p["match"][k] is not None])
        if len(w):
            print(f"{k:8s} {np.median(r):10.2f} / {np.percentile(r, 95):5.2f} {np.median(w):12.2f} / {w.max():5.2f}")
    print("\ncandidate flag   rows flagged   of which wrong   (thresholds leave-one-section-out)")
    counts = Counter()
    for f in files:
        th = thresholds_from([r for g in files if g != f for r in data[g]])
        for t, p in data[f]:
            if named(p):
                for fl in candidate_flags(p["match"], th):
                    counts[fl, "n"] += 1
                    counts[fl, "wrong"] += not same(t, p)
    for fl in ("poor_shape", "poor_colour", "near_tie", "prior"):
        print(f"{fl:14s} {counts[fl, 'n']:12d} {counts[fl, 'wrong']:16d}")


if __name__ == "__main__":
    main()
