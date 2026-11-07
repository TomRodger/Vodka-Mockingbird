"""
Decide which texture descriptor to use for clustering lithology intervals.

Run from the repo root:

    python -m tools.texture_test                       # every tools/labels/*.csv
    python -m tools.texture_test tools/labels/A.csv tools/labels/B.csv
    python -m tools.texture_test --wins 64 32 16 --seeds 0 1 2   # (these are the defaults)

Label CSVs are made by label_tool.py `check` (columns top_y, bottom_y, lithology;
file name STEM__y<row>.csv tells the script which image it belongs to).

What it does, for every labelled section x window height x seed:
  1. samples up to --per-interval windows inside each interval. Windows are
     WIN rows tall and as wide as the column allows (up to WIN), because most
     columns are narrower than 64 px.
  2. describes every window with each texture descriptor, and with each
     texture descriptor + fill colour.
  3. scores how well those descriptions separate the true lithologies.

Descriptors
  fft, hog_pooled, hog_concat   see features.py
  fft+hog_pooled                average of the two (normalised) distances
  "+colour" rows                distance = max(texture, colour / TAU_C):
                                two windows are close only if BOTH agree

Metrics (higher is better unless noted)
  ari_int      main metric. Each interval = mean of its windows; average-linkage
               clustering cut at the true number of lithologies; Adjusted Rand
               Index vs the truth (1 = perfect, ~0 = random). Optimistic: it is
               told the true number of clusters.
  silhouette   separation margin of the true classes (-1..1). The tie-breaker.
  nn_acc       nearest window from a DIFFERENT interval has the same lithology.
  ari_win      like ari_int but on individual windows.
  shift_ratio  (lower is better) how much moving a window 3-8 px changes its
               texture, relative to the gap between different lithologies.
  ms_per_win   feature time per window (texture + colour).

Output: a per-run table in out/texture_results.csv, a summary per setting,
and the verdict from the decision rule fixed in README.md.
"""

import argparse
import csv
import glob
import json
import os
import sys
import time

import numpy as np
from PIL import Image
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import cdist, squareform
from sklearn.metrics import adjusted_rand_score, silhouette_score

from lithology.data import DATASET_ANNOTATIONS, DATASET_IMAGES, OUT_DIR

from .texture_features import TEXTURE_DESCRIPTORS, colour_feat

HERE = os.path.dirname(os.path.abspath(__file__))

Image.MAX_IMAGE_PIXELS = None

IMAGES_DIR = DATASET_IMAGES
ANN_DIR = DATASET_ANNOTATIONS
TRIM = 4          # px shaved off every interval edge (boundary lines, borders)
TAU_C = 10.0      # colour distance (Lab, Delta-E) treated as "one unit" of difference
COMBOS = {"fft+hog_pooled": ("fft", "hog_pooled")}
NOISE_FLOOR = 0.01


# ---------------------------------------------------------------- data

def find_image(stem):
    hits = [p for p in glob.glob(f"{IMAGES_DIR}/*") if os.path.splitext(os.path.basename(p))[0] == stem]
    if not hits:
        sys.exit(f"No image for {stem} in {IMAGES_DIR}")
    return hits[0]


def load_section(labels_path):
    base = os.path.splitext(os.path.basename(labels_path))[0]
    if "__y" not in base:
        sys.exit(f"{labels_path}: name must be STEM__y<row>.csv (made by label_tool.py check)")
    stem = base.rsplit("__y", 1)[0]
    box = json.load(open(f"{ANN_DIR}/{stem}.json"))["bounding_boxes"]["interpreted_lithology_column"]
    rows = list(csv.DictReader(open(labels_path, newline="")))
    iv = [(int(r["top_y"]), int(r["bottom_y"]), r["lithology"].strip()) for r in rows]
    ya, yb = min(a for a, _, _ in iv), max(b for _, b, _ in iv)
    crop = Image.open(find_image(stem)).crop((box[0], ya, box[2], yb)).convert("RGB")
    col = np.asarray(crop, dtype=np.float32) / 255.0
    iv = [x for x in iv if x[2].upper() != "UNSURE"]       # labelled UNSURE = skipped
    return base, col, [(a - ya, b - ya, n) for a, b, n in iv]


def sample(col, intervals, win, per_interval, rng):
    H, W = col.shape[:2]
    ww = min(win, W - 2 * TRIM)
    if ww < 16:
        return None
    x_lo, x_hi = TRIM, W - TRIM - ww
    wins, twins, labels, groups = [], [], [], []
    for i, (top, bot, lith) in enumerate(intervals):
        y_lo, y_hi = top + TRIM, bot - TRIM - win
        if y_hi < y_lo:
            continue
        for _ in range(per_interval):
            y, x = rng.integers(y_lo, y_hi + 1), rng.integers(x_lo, x_hi + 1)
            wins.append(col[y:y + win, x:x + ww])
            dy, dx = rng.integers(3, 9, size=2) * rng.choice([-1, 1], size=2)
            ty, tx = int(np.clip(y + dy, y_lo, y_hi)), int(np.clip(x + dx, x_lo, x_hi))
            twins.append(col[ty:ty + win, tx:tx + ww])
            labels.append(lith)
            groups.append(i)
    return ww, wins, twins, np.array(labels), np.array(groups)


# ---------------------------------------------------------------- metrics

def scores(D, Di, labels, groups, li, twin_ratio):
    k = len(set(labels))
    out = dict(nn_acc=np.nan, silhouette=np.nan, ari_win=np.nan, ari_int=np.nan,
               shift_ratio=twin_ratio)
    if k < 2:
        return out
    Dn = D.copy()
    Dn[groups[:, None] == groups[None, :]] = np.inf
    if np.isfinite(Dn).any(axis=1).all():
        out["nn_acc"] = float(np.mean(labels[Dn.argmin(axis=1)] == labels))
    if len(labels) > k:
        out["silhouette"] = float(silhouette_score(D, labels, metric="precomputed"))
        out["ari_win"] = adjusted_rand_score(
            labels, fcluster(linkage(squareform(D, checks=False), "average"), k, "maxclust"))
    if len(li) > k:
        out["ari_int"] = adjusted_rand_score(
            li, fcluster(linkage(squareform(Di, checks=False), "average"), k, "maxclust"))
    return out


def run_one(col, intervals, win, seed, per_interval):
    rng = np.random.default_rng(seed)
    s = sample(col, intervals, win, per_interval, rng)
    if s is None or len(s[1]) == 0:
        return None, None
    ww, wins, twins, labels, groups = s
    ids = np.unique(groups)
    li = np.array([labels[groups == g][0] for g in ids])
    meta = dict(win_w=ww, n_int_used=len(ids), n_int_total=len(intervals), k=len(set(labels)))

    def interval_means(F):
        return np.array([F[groups == g].mean(axis=0) for g in ids])

    t0 = time.perf_counter()
    C = np.array([colour_feat(w) for w in wins])
    t_col = (time.perf_counter() - t0) / len(wins) * 1000
    Dc, Dci = cdist(C, C) / TAU_C, cdist(interval_means(C), interval_means(C)) / TAU_C

    tex = {}   # name -> (D, Di, twin distances, ms)
    for name, fn in TEXTURE_DESCRIPTORS.items():
        t0 = time.perf_counter()
        F = np.array([fn(w) for w in wins])
        ms = (time.perf_counter() - t0) / len(wins) * 1000
        Ft = np.array([fn(w) for w in twins])
        D = cdist(F, F)
        scale = np.median(D[np.triu_indices(len(D), 1)]) or 1.0   # unsupervised unit
        Fi = interval_means(F)
        tex[name] = (D / scale, cdist(Fi, Fi) / scale, np.linalg.norm(F - Ft, axis=1) / scale, ms)
    for name, (a, b) in COMBOS.items():
        A, B = tex[a], tex[b]
        tex[name] = ((A[0] + B[0]) / 2, (A[1] + B[1]) / 2, (A[2] + B[2]) / 2, A[3] + B[3])

    between = labels[:, None] != labels[None, :]
    rows = []
    for name, (D, Di, twin, ms) in tex.items():
        gap = np.median(D[between]) if between.any() else 0.0
        ratio = float(np.median(twin) / gap) if gap > 0 else np.nan   # nan: texture empty (e.g. no linework)
        rows.append(dict(descriptor=name, colour=False, ms_per_win=ms,
                         **scores(D, Di, labels, groups, li, ratio)))
        rows.append(dict(descriptor=name, colour=True, ms_per_win=ms + t_col,
                         **scores(np.maximum(D, Dc), np.maximum(Di, Dci), labels, groups, li, ratio)))
    return meta, rows


# ---------------------------------------------------------------- summary + verdict

def verdict(summary):
    """Decision rule (fixed in advance, see README):
    1. keep every descriptor whose mean ari_int is within 2x seed noise of the best;
       if only one is left, it wins on ari_int
    2. among those, keep every one whose silhouette is within 2x noise of the best;
       if only one is left, it wins on silhouette
    3. otherwise the fastest of those wins."""
    cands = [s for s in summary if not np.isnan(s["ari_int"])] or summary
    reasons = []
    for m in ("ari_int", "silhouette"):
        vals = [s for s in cands if not np.isnan(s[m])]
        if not vals:
            continue
        best = max(vals, key=lambda s: s[m])
        noise = 2 * max(max(s[m + "_noise"] for s in vals), NOISE_FLOOR)
        keep = [s for s in vals if s[m] >= best[m] - noise]
        if len(keep) == 1:
            rest = sorted((s for s in vals if s is not best), key=lambda s: -s[m])
            runner = f" vs {rest[0]['descriptor']} {rest[0][m]:.3f}" if rest else ""
            return best["descriptor"], " -> ".join(reasons + [
                f"{m}: {best[m]:.3f}{runner} (gap > noise {noise:.3f})"])
        reasons.append(f"{m} ties within {noise:.3f}: {', '.join(s['descriptor'] for s in keep)}")
        cands = keep
    fastest = min(cands, key=lambda s: s["ms_per_win"])
    return fastest["descriptor"], " -> ".join(reasons + [f"fastest of those ({fastest['ms_per_win']:.2f} ms/win)"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("labels", nargs="*", help="label CSVs (default: tools/labels/*.csv)")
    ap.add_argument("--wins", type=int, nargs="+", default=[64, 32, 16], help="window heights in px")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--per-interval", type=int, default=25)
    args = ap.parse_args()

    paths = args.labels or sorted(glob.glob(os.path.join(HERE, "labels", "*.csv")))
    if not paths:
        sys.exit("No label CSVs found. Make some with label_tool.py (see README).")

    results = []
    for p in paths:
        name, col, intervals = load_section(p)
        print(f"\n== {name}  (column {col.shape[1]} px wide, {len(intervals)} intervals, "
              f"{len(set(n for _, _, n in intervals))} lithologies)")
        for win in args.wins:
            for seed in args.seeds:
                meta, rows = run_one(col, intervals, win, seed, args.per_interval)
                if meta is None:
                    print(f"   win {win}: no usable windows (intervals or column too small)")
                    break
                if seed == args.seeds[0]:
                    print(f"   win {win}: windows {win}x{meta['win_w']}, "
                          f"{meta['n_int_used']}/{meta['n_int_total']} intervals tall enough, "
                          f"{meta['k']} lithologies among them")
                for r in rows:
                    results.append(dict(section=name, win=win, seed=seed, **meta, **r))

    os.makedirs(OUT_DIR, exist_ok=True)
    out_csv = os.path.join(OUT_DIR, "texture_results.csv")
    keys = list(results[0].keys())
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(results)

    metrics = ["ari_int", "silhouette", "nn_acc", "ari_win", "shift_ratio", "ms_per_win"]
    descs = list(dict.fromkeys(r["descriptor"] for r in results))
    sections = list(dict.fromkeys(r["section"] for r in results))
    print(f"\nPer-run results: {out_csv}")
    print("Values are means over sections (each section first averaged over seeds); "
          "(±) is the typical spread across seeds.")
    for win in args.wins:
        for colour in (False, True):
            summary = []
            for d in descs:
                rs = [r for r in results if r["win"] == win and r["colour"] == colour and r["descriptor"] == d]
                if not rs:
                    continue
                s = dict(descriptor=d)
                for m in metrics:
                    per_sec, noise = [], []
                    for sec in sections:
                        v = [r[m] for r in rs if r["section"] == sec and not np.isnan(r[m])]
                        if v:
                            per_sec.append(np.mean(v))
                            noise.append(np.std(v))
                    s[m] = float(np.mean(per_sec)) if per_sec else np.nan
                    s[m + "_noise"] = float(np.mean(noise)) if noise else 0.0
                summary.append(s)
            if not summary:
                continue
            for s in summary:
                s["wins"] = 0
            for sec in sections:
                vals = {s["descriptor"]: np.nanmean([r["ari_int"] for r in results if r["win"] == win
                                                     and r["colour"] == colour and r["section"] == sec
                                                     and r["descriptor"] == s["descriptor"]] or [np.nan])
                        for s in summary}
                vals = {k: v for k, v in vals.items() if not np.isnan(v)}
                if vals:
                    top = max(vals.values())
                    for s in summary:
                        if vals.get(s["descriptor"], -9) >= top - 1e-9:
                            s["wins"] += 1
            n_sec = sum(1 for sec in sections if any(r["section"] == sec and r["win"] == win and not np.isnan(r["ari_int"])
                                                    for r in results))
            label = "texture + colour" if colour else "texture only"
            print(f"\n--- window height {win}, {label} ---")
            print(f"{'descriptor':>15s} {'ari_int':>13s} {'silhouette':>13s} {'nn_acc':>7s} {'ari_win':>7s} "
                  f"{'shift':>6s} {'ms/win':>7s} {'best on':>8s}")
            for s in summary:
                print(f"{s['descriptor']:>15s} {s['ari_int']:7.3f}±{s['ari_int_noise']:.3f} "
                      f"{s['silhouette']:7.3f}±{s['silhouette_noise']:.3f} {s['nn_acc']:7.3f} {s['ari_win']:7.3f} "
                      f"{s['shift_ratio']:6.3f} {s['ms_per_win']:7.2f} {s['wins']:>3d}/{n_sec}")
            winner, why = verdict(summary)
            print(f"VERDICT: {winner}  -- {why}")
            ari = [s["ari_int"] for s in summary if not np.isnan(s["ari_int"])]
            if ari and min(ari) > 0.99:
                print("NOTE: ari_int is ~1.0 for every descriptor: these sections are too easy to "
                      "separate them on ari_int, so silhouette decided. Add harder sections.")
            elif colour and ari:
                spread = max(ari) - min(ari)
                print(f"Texture choice once colour is added: ari_int spread across descriptors = {spread:.3f}"
                      + ("  (small: colour is doing most of the work)" if spread < 0.05 else ""))


if __name__ == "__main__":
    main()
