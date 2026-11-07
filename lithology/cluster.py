"""Stage 2 - clustering: which intervals are the same rock?

    python -m lithology.cluster IMAGE ANNOTATION_JSON [--intervals CSV] [--truth CSV]
                                [--y0 ROW --y1 ROW] [--texture pooled|team] [--out DIR]

    IMAGE / ANNOTATION_JSON: a scan and its annotation file
    --intervals: CSV with top_y,bottom_y (e.g. example.csv). If omitted, stage 1
                 (lithology.segment) is run on the lithology column (rows y0..y1).
    --truth:     CSV with top_y,bottom_y,lithology, for scoring (see lithology.evaluate).

Output (in out/ by default): per-interval CSV with cluster_id / margin / flags, merged CSV,
representatives.json (the handoff to legend matching), overlay and montage images.
The legacy --texture team option is an alias for the shared pooled FFT.

How it clusters (each choice is explained where it happens below):
  1. describe each interval: Lab fill colour + texture (from the shared FeatureExtractor's
     pattern mask), with guards for thin intervals, patternless fills and blank paper
  2. distance = max(colour / TAU_C, texture / noise unit): close only if BOTH agree
  3. average-linkage clustering of intervals tall enough for texture, cut at 1.0
  4. thin intervals: attached by colour; if colour can't decide, by their neighbours
  5. adjacent intervals in the same cluster are merged, then re-described and re-clustered
"""

import argparse
import json
import os
import warnings

import cv2
import numpy as np
from PIL import Image
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import cdist, squareform

from .data import OUT_DIR, load_annotation, read_intervals
from .features import FeatureExtractor, bgr_to_lab
from .output import save_montage, save_overlay, write_csv
from .segment import build_intervals, detect_boundaries, section_edge_margins

# ---------------------------------------------------------------- defaults
# Starting points, NOT tuned values. Tune them on hand-labelled sections (row ARI).
TRIM = 4             # px removed from every patch edge: boundary lines and column borders
MIN_TEXTURE_H = 24   # intervals shorter than this are "thin": too few rows for texture, so they
                     # attach by colour, or by their neighbours when colour can't decide.
                     # (64 was tried: on the old segmentation the only tall pieces were mixed.)
MAX_ROUNDS = 3       # re-describe merged intervals and re-cluster, until nothing merges
MIN_CONTRAST = 40    # grey-level spread (1st-99th percentile) below this = no pattern drawn
TAU_C = 10.0         # Lab Delta-E counted as one unit; ~10 is a clearly visible difference
NOISE_K = 3.0        # texture counts as different only above NOISE_K x this document's noise
NOISE_NEIGHBOUR = 3  # noise = distance to each interval's 3rd-nearest neighbour (see texture_unit)
TAU_T_FLOOR = {"pooled": 0.01, "team": 0.01}  # team is a compatibility alias for pooled
CUT = 1.0            # merge clusters while their average distance is below this
LOW_MARGIN = 0.7     # flag intervals this far (in units) from their cluster's centre
BLANK_L = 95         # a patternless interval this light (Lab L*) and grey is blank paper:
BLANK_AB = 5         #   no rock logged there. Output as cluster_id -1, flag "no_data".


# ---------------------------------------------------------------- 1. describe intervals

def describe(col, col_y0, top, bottom, fe, texture):
    """Features for one interval. col is the lithology column as RGB, starting at row col_y0."""
    p = col[top - col_y0 + TRIM: bottom - col_y0 - TRIM, TRIM: col.shape[1] - TRIM]
    if p.shape[0] < 2:                      # too thin to trim: use it untrimmed
        p = col[top - col_y0: bottom - col_y0, TRIM: col.shape[1] - TRIM]
    bgr = np.ascontiguousarray(p[:, :, ::-1])   # the shared extractor expects BGR (cv2.imread order)
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")          # extractor warns when a patch has no background pixels
        f = fe.extract(bgr)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    lo, hi = np.percentile(gray, [1, 99])
    has_pattern = (hi - lo) >= MIN_CONTRAST

    # Colour: the extractor's median of non-pattern pixels, except when there is no real
    # pattern (Otsu then splits noise, so use every pixel) or no background pixels at all.
    background = (f["pattern_mask"] == 0).any()
    fill = f["median_colour"] if (has_pattern and background) else np.median(bgr.reshape(-1, 3), axis=0)

    height = bottom - top
    if height < MIN_TEXTURE_H:
        tex = None                                       # thin: handled in the second pass
    elif not has_pattern:
        tex = "empty"                                    # solid fill: a texture of its own
    else:
        # The shared extractor now supplies pooled FFT for both column
        # intervals and legend swatches. Do not calculate a second descriptor.
        tex = f["fft_features"]

    mid = max(0, p.shape[0] // 2 - 20)
    n_tiles = max(1, p.shape[0] // 32)          # how many texture samples the estimate averages
    lab = bgr_to_lab(fill)
    # Blank test on the INNER part only: thick column borders and a border line at the
    # bottom of the log sit inside the patch and would otherwise read as "pattern".
    ih, iw = gray.shape
    inner = gray[ih // 10: ih - ih // 10, iw // 5: iw - iw // 5]
    blank = False
    if inner.size:
        ilo, ihi = np.percentile(inner, [1, 99])
        inner_bgr = bgr[ih // 10: ih - ih // 10, iw // 5: iw - iw // 5].reshape(-1, 3)
        ilab = bgr_to_lab(np.median(inner_bgr, axis=0))
        blank = (ihi - ilo) < MIN_CONTRAST and ilab[0] >= BLANK_L and \
            abs(ilab[1]) < BLANK_AB and abs(ilab[2]) < BLANK_AB
    return dict(top=top, bottom=bottom, lab=lab, tex=tex, n_tiles=n_tiles, blank=blank,
                thin=tex is None, no_pattern=not has_pattern,
                thumb=p[mid: mid + 40].copy())


# ---------------------------------------------------------------- 2-4. cluster

def texture_matrix(descs, idx, texture):
    """Texture vectors for core intervals; 'empty' (no pattern) becomes all zeros, which is far
    from any drawn pattern and identical to other solid fills."""
    dim = next((len(descs[i]["tex"]) for i in idx if not isinstance(descs[i]["tex"], str)),
               24)
    return np.array([np.zeros(dim) if isinstance(descs[i]["tex"], str) else descs[i]["tex"] for i in idx])


def texture_unit(Dt, texture):
    """How big a texture difference counts as 'one unit', measured per document.

    Texture noise varies a lot between documents (narrow columns and short intervals give
    noisier spectra), so a fixed threshold fails: one tuned for the example shatters Rio
    Nueces' shale. The distance from each interval to its 3rd-nearest neighbour is mostly
    same-rock noise, so the unit is NOISE_K x the median of those.
      - 1st-nearest was too small: in Rio almost every interval has a near-duplicate twin,
        so it measured duplicates, not typical spread (15 clusters for ~2 rocks; 3rd: 10).
      - 4th-nearest merged two of the example's rocks: sandstone has only 4 intervals, so
        its 4th neighbour is another rock. The neighbour must stay below the smallest
        real group's size.
    Unlike dividing by the overall median, this does NOT inflate noise in a column that
    is all one rock."""
    if len(Dt) < 2:
        return TAU_T_FLOOR[texture]
    D = np.sort(Dt, axis=1)[:, 1:]              # drop each interval's distance to itself
    k = min(NOISE_NEIGHBOUR, D.shape[1]) - 1
    return max(TAU_T_FLOOR[texture], NOISE_K * float(np.median(D[:, k])))


def cluster(descs, texture):
    """Cluster every interval except blank ones (no rock logged), which get cluster -1.
    Blank paper must not become a 'rock type' that legend matching then names."""
    keep = [i for i, d in enumerate(descs) if not d["blank"]]
    n = len(descs)
    labels, margin = np.full(n, -1), np.zeros(n)
    flags = [{"no_data"} for _ in range(n)]
    if keep:
        lk, mk, fk = _cluster([descs[i] for i in keep], texture)
        for j, i in enumerate(keep):
            labels[i], margin[i], flags[i] = lk[j], mk[j], fk[j]
    return labels, margin, flags


def _cluster(descs, texture):
    n = len(descs)
    labels = np.full(n, -1)
    lab = np.array([d["lab"] for d in descs])
    Dc = cdist(lab, lab) / TAU_C
    core = [i for i, d in enumerate(descs) if not d["thin"]]
    flags = [set() for _ in range(n)]
    for i, d in enumerate(descs):
        if d["thin"]:
            flags[i].add("thin")
        if d["no_pattern"]:
            flags[i].add("no_pattern")

    # Core intervals: combined distance. max(), not a sum or a long concatenated vector:
    # two intervals are only "the same" if colour AND texture both agree.
    D = None
    if len(core) >= 2:
        T = texture_matrix(descs, core, texture)
        # A short interval's texture is averaged over few 32-row tiles, so it is noisier.
        # Scale each pair's distance by its expected noise, sqrt(1/n_i + 1/n_j): short
        # intervals get more tolerance, tall ones (reliable texture) less.
        nt = np.array([descs[i]["n_tiles"] for i in core], dtype=float)
        Dt = cdist(T, T) / np.sqrt(1 / nt[:, None] + 1 / nt[None, :])
        D = np.maximum(Dc[np.ix_(core, core)], Dt / texture_unit(Dt, texture))
        # Average linkage: merges groups by their average distance, so a chain of
        # slightly-similar intervals can't glue two different rocks together.
        labels[core] = fcluster(linkage(squareform(D, checks=False), "average"), CUT, "distance")
    elif len(core) == 1:
        labels[core] = 1

    # Thin intervals: no texture, so colour decides. If several clusters share that colour
    # (always true in black-and-white logs), the nearest labelled neighbour decides: thin
    # pieces are usually fragments of the rock around them (over-segmentation).
    cents = {c: np.median(lab[labels == c], axis=0) for c in set(labels[labels > 0])}
    next_id = max(cents, default=0) + 1
    thin_dist = np.zeros(n)
    for i in [i for i in range(n) if descs[i]["thin"]]:
        dist = {c: np.linalg.norm(lab[i] - m) / TAU_C for c, m in cents.items()}
        cands = [c for c, v in dist.items() if v < CUT]
        if not cands:
            labels[i] = next_id
            cents[next_id] = lab[i]
            next_id += 1
            flags[i].add("new_from_thin")
        elif len(cands) == 1:
            labels[i] = cands[0]
        else:
            labels[i] = neighbour_label(labels, i, cands) or min(cands, key=dist.get)
            flags[i].add("colour_ambiguous")
        thin_dist[i] = dist.get(labels[i], 0.0)

    # Margin: distance to the cluster's medoid (the member with the smallest total distance
    # to the others). Near 0 = typical member, near 1 = borderline.
    margin = thin_dist.copy()
    if D is not None:
        cl = labels[core]
        for c in set(cl):
            m = np.where(cl == c)[0]
            med = m[np.argmin(D[np.ix_(m, m)].sum(axis=1))]
            for j in m:
                margin[core[j]] = D[j, med]
    for i in range(n):
        if margin[i] > LOW_MARGIN:
            flags[i].add("low_margin")

    # Renumber 0..k-1 by first appearance from the top (like example.csv).
    order = {}
    for c in labels:
        order.setdefault(c, len(order))
    return np.array([order[c] for c in labels]), margin, flags


def neighbour_label(labels, i, cands):
    """Label of the nearest interval above or below whose cluster is in cands."""
    for k in range(1, len(labels)):
        for j in (i - k, i + k):
            if 0 <= j < len(labels) and labels[j] in cands:
                return labels[j]
    return None


# ---------------------------------------------------------------- 5. merge

def merge(descs, labels, margin, flags):
    """Join touching intervals that landed in the same cluster. With an over-segmented input
    this removes the false boundaries; a real boundary always separates different clusters."""
    order = np.argsort([d["top"] for d in descs])
    out = []
    for i in order:
        d = descs[i]
        if out and out[-1]["cluster_id"] == labels[i] and abs(d["top"] - out[-1]["bottom_y"]) <= 1:
            o = out[-1]
            o["bottom_y"] = d["bottom"]
            o["n_pieces"] += 1
            o["margin"] = max(o["margin"], margin[i])
            o["flags"] |= flags[i]
        else:
            out.append(dict(top_y=d["top"], bottom_y=d["bottom"], cluster_id=int(labels[i]),
                            n_pieces=1, margin=float(margin[i]), flags=set(flags[i])))
    return out


def representatives(descs, labels, margin):
    """Handoff for legend matching: the most typical member (lowest margin) of each cluster."""
    reps = []
    for c in range(labels.max() + 1):
        m = [i for i in range(len(descs)) if labels[i] == c]
        core = [i for i in m if not descs[i]["thin"]] or m
        r = min(core, key=lambda i: margin[i])
        reps.append(dict(cluster_id=c, n_members=len(m),
                         total_rows=int(sum(descs[i]["bottom"] - descs[i]["top"] for i in m)),
                         representative=dict(top_y=descs[r]["top"], bottom_y=descs[r]["bottom"]),
                         lab_colour=[round(float(v), 2) for v in descs[r]["lab"]]))
    return reps



# ---------------------------------------------------------------- run one document

def run(image, bbox, intervals_csv=None, truth_csv=None, y0=None, y1=None, texture="pooled",
        out_dir=None, tag=None, quiet=False):
    box = load_annotation(bbox)["interpreted_lithology_column"]
    stem = os.path.splitext(os.path.basename(image))[0]
    im = Image.open(image)

    if intervals_csv:
        intervals = read_intervals(intervals_csv)
        source = os.path.basename(intervals_csv)
    else:   # stage 1 (lithology.segment) on the requested rows of the lithology column
        ya, yb = max(y0 or box[1], box[1]), min(y1 or box[3], box[3])
        crop = im.crop((box[0], ya, box[2], yb))
        margins = section_edge_margins((box[0], ya, box[2], yb), box)
        intervals = build_intervals(detect_boundaries(crop, edge_margin=margins), ya)
        source = "segmentation"
    ya = min(r["top_y"] for r in intervals)
    yb = max(r["bottom_y"] for r in intervals)
    col = np.asarray(im.crop((box[0], ya, box[2], yb)).convert("RGB"))   # only these rows in memory

    # Cluster, merge, then re-describe the merged intervals and cluster again. Merging turns
    # fragments into tall intervals whose texture is reliable, so later rounds see better
    # features. Stops when a round merges nothing.
    fe = FeatureExtractor()
    cur = [dict(top_y=r["top_y"], bottom_y=r["bottom_y"]) for r in intervals]
    for rnd in range(MAX_ROUNDS):
        descs = [describe(col, ya, r["top_y"], r["bottom_y"], fe, texture) for r in cur]
        labels, margin, flags = cluster(descs, texture)
        merged = merge(descs, labels, margin, flags)
        if len(merged) == len(cur):
            break
        cur = [dict(top_y=m["top_y"], bottom_y=m["bottom_y"]) for m in merged]
    rounds = rnd + 1

    # Map every ORIGINAL input interval to the final merged interval that contains it.
    starts = [m["top_y"] for m in merged]
    for m in merged:
        m["n_pieces"] = 0
    raw = []
    for r in intervals:
        m = merged[max(0, np.searchsorted(starts, r["top_y"], side="right") - 1)]
        m["n_pieces"] += 1
        raw.append(dict(top_y=r["top_y"], bottom_y=r["bottom_y"], cluster_id=m["cluster_id"],
                        margin=round(m["margin"], 3), flags=m["flags"]))

    out_dir = out_dir or OUT_DIR
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, f"{stem}__{tag or texture}")
    write_csv(base + "_intervals.csv", raw, ["top_y", "bottom_y", "cluster_id", "margin", "flags"])
    for r in merged:
        r["margin"] = round(r["margin"], 3)
    write_csv(base + "_merged.csv", merged, ["top_y", "bottom_y", "cluster_id", "n_pieces", "margin", "flags"])
    json.dump(representatives(descs, labels, margin), open(base + "_representatives.json", "w"), indent=2)
    save_overlay(base + "_overlay.png", col, ya, merged, int(labels.max() + 1))
    save_montage(base + "_montage.png", descs, labels, margin)

    res = dict(source=source, texture=texture, n_clusters=int(labels.max() + 1), rounds=rounds,
               n_no_data=int(sum(r["cluster_id"] < 0 for r in merged)),
               n_thin=sum(d["thin"] for d in descs), n_low_margin=sum("low_margin" in f for f in flags))
    if truth_csv:
        from .evaluate import score          # imported here: evaluate imports this module
        res.update(score(read_intervals(truth_csv), merged, len(intervals)))
    if not quiet:
        print(f"{stem}: {len(intervals)} intervals from {source} -> {res['n_clusters']} clusters, "
              f"{len(merged)} after merging in {rounds} round(s) ({res['n_thin']} thin, "
              f"{res['n_low_margin']} low-margin, {res['n_no_data']} no-data)")
        if truth_csv:
            print(f"  row ARI {res['row_ari']:.3f} | boundaries recall {res['b_recall']:.2f} "
                  f"precision {res['b_precision']:.2f} | true intervals {res['n_true']}")
        print(f"  outputs: {base}_*")
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image")
    ap.add_argument("bbox")
    ap.add_argument("--intervals", help="CSV with top_y,bottom_y; default: run lithology.segment")
    ap.add_argument("--truth", help="CSV with top_y,bottom_y,lithology for scoring")
    ap.add_argument("--y0", type=int)
    ap.add_argument("--y1", type=int)
    ap.add_argument("--texture", choices=["pooled", "team"], default="pooled",
                    help="pooled FFT; team is a compatibility alias using the same descriptor")
    ap.add_argument("--out")
    a = ap.parse_args()
    run(a.image, a.bbox, a.intervals, a.truth, a.y0, a.y1, a.texture, a.out)


if __name__ == "__main__":
    main()
