"""
Cluster lithology intervals by visual pattern (task: "group intervals containing
the same or similar visual patterns").

Pipeline position:
    backend/app/task1.py      -> intervals [{"top_y", "bottom_y"}] (full-image rows)
    clustering/cluster.py     -> cluster_id + confidence per interval, merged intervals,
                                 one representative per cluster
    legend matching           -> a name per cluster (uses the same FeatureExtractor)

Run from the repo root:

    python clustering/cluster.py --example
        both checks on the organisers' example, for both texture options

    python clustering/cluster.py IMAGE BBOX_JSON [--intervals CSV] [--truth CSV]
                                 [--y0 ROW --y1 ROW] [--texture pooled|team] [--out DIR]
        IMAGE / BBOX_JSON: a log and its annotation file
        --intervals: CSV with top_y,bottom_y (e.g. example.csv). If omitted, task1.py's
                     segmentation is run on the lithology column (rows y0..y1).
        --truth:     CSV with top_y,bottom_y,lithology, for scoring.

How it clusters (each choice is explained where it happens below):
  1. describe each interval: Lab fill colour + texture (from the shared FeatureExtractor's
     pattern mask), with guards for thin intervals and patternless fills
  2. distance = max(colour / TAU_C, texture / TAU_T): close only if BOTH agree
  3. average-linkage clustering of intervals tall enough for texture, cut at 1.0
  4. thin intervals: attached by colour; if colour can't decide, by their neighbours
  5. adjacent intervals in the same cluster are merged
"""

import argparse
import csv
import json
import os
import sys
import warnings

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import cdist, squareform
from sklearn.metrics import adjusted_rand_score

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "extracting_pattern"))
sys.path.insert(0, os.path.join(REPO, "backend", "app"))
from feature_extractor import FeatureExtractor  # noqa: E402  (teammate's shared extractor)
from task1 import build_intervals, detect_boundaries  # noqa: E402  (teammate's segmentation)

Image.MAX_IMAGE_PIXELS = None

# ---------------------------------------------------------------- defaults
# Starting points, NOT tuned values. Tune them on hand-labelled sections (row ARI).
TRIM = 4             # px removed from every patch edge: boundary lines and column borders
MIN_TEXTURE_H = 24   # intervals shorter than this are "thin": too few rows for texture, so they
                     # attach by colour, or by their neighbours when colour can't decide.
                     # (64 was tried: on task1 output the only tall pieces are the mixed ones.)
MAX_ROUNDS = 3       # re-describe merged intervals and re-cluster, until nothing merges
MIN_CONTRAST = 40    # grey-level spread (1st-99th percentile) below this = no pattern drawn
TAU_C = 10.0         # Lab Delta-E counted as one unit; ~10 is a clearly visible difference
NOISE_K = 3.0        # texture counts as different only above NOISE_K x this document's noise
NOISE_NEIGHBOUR = 3  # noise = distance to each interval's 3rd-nearest neighbour (see texture_unit)
TAU_T_FLOOR = {"pooled": 0.01, "team": 0.5}   # lower bound on the texture unit
CUT = 1.0            # merge clusters while their average distance is below this
LOW_MARGIN = 0.7     # flag intervals this far (in units) from their cluster's centre
BLANK_L = 95         # a patternless interval this light (Lab L*) and grey is blank paper:
BLANK_AB = 5         #   no rock logged there. Output as cluster_id -1, flag "no_data".
EXAMPLE = os.path.join(REPO, "hack27", "project", "example")
EXAMPLE_STEM = "0000000009_d-582_0000020435-0001_log06_fe_5in_md"


# ---------------------------------------------------------------- texture: pooled FFT

_grid = {}


def pooled_fft(mask, nr=6, na=4):
    """FFT texture pooled into spacing (radial) x orientation (angle) bins.

    Computed on the shared extractor's pattern mask. Unlike the 16x16 spectrum it:
      - averages over 32-row tiles, so tall intervals aren't one blurry spectrum
      - applies a Hann window, so tile edges don't add fake frequencies
      - pools into 24 bins in cycles/pixel, which mean the same for any patch size
    """
    m = mask.astype(np.float32)
    h, w = m.shape
    th = 32 if h >= 32 else h
    specs = []
    for y in range(0, h - th + 1, th):
        t = m[y:y + th]
        key = (th, w)
        if key not in _grid:
            fy = (np.arange(th) - th // 2) / th
            fx = (np.arange(w) - w // 2) / w
            FY, FX = np.meshgrid(fy, fx, indexing="ij")
            rb = np.minimum((np.hypot(FY, FX) / 0.5 * nr).astype(int), nr - 1)
            ab = np.minimum(((np.arctan2(FY, FX) % np.pi) / np.pi * na).astype(int), na - 1)
            _grid[key] = ((rb * na + ab).ravel(), np.outer(np.hanning(th), np.hanning(w)))
        bins, hann = _grid[key]
        S = np.abs(np.fft.fftshift(np.fft.fft2((t - t.mean()) * hann)))
        specs.append(np.bincount(bins, weights=S.ravel(), minlength=nr * na))
    f = np.mean(specs, axis=0)
    s = f.sum()
    return f / s if s > 0 else f


# ---------------------------------------------------------------- 1. describe intervals

def bgr_to_lab(bgr):
    """OpenCV 8-bit Lab -> real Lab (L 0-100, a/b around 0), so Delta-E is in standard units."""
    L, a, b = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2LAB)[0, 0].astype(float)
    return np.array([L * 100 / 255, a - 128, b - 128])


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
    elif texture == "team":
        tex = f["fft_features"]
    else:
        tex = pooled_fft(f["pattern_mask"] > 0)

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
               24 if texture == "pooled" else 256)
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


# ---------------------------------------------------------------- outputs

def font(size):
    for name in ("DejaVuSans.ttf", "Arial.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def palette(k):
    import colorsys
    return [tuple(int(255 * c) for c in colorsys.hsv_to_rgb((i * 0.618034) % 1, 0.7, 0.9)) for i in range(k)]


def write_csv(path, rows, keys):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("|".join(sorted(r[k])) if isinstance(r[k], set) else r[k]) for k in keys})


def save_overlay(path, col, col_y0, merged, k, max_h=6000):
    """Lithology column with a cluster-coloured bar and boundary lines (scaled to fit)."""
    H, W = col.shape[:2]
    s = min(max(1, round(200 / W)), max_h / H)
    img = Image.fromarray(col).resize((max(1, round(W * s)), max(1, round(H * s))), Image.Resampling.NEAREST)
    canvas = Image.new("RGB", (img.width + 150, img.height), "white")
    canvas.paste(img, (0, 0))
    d = ImageDraw.Draw(canvas)
    cols, f = palette(max(k, 1)), font(12)
    for r in merged:
        a, b = (r["top_y"] - col_y0) * s, (r["bottom_y"] - col_y0) * s
        c = r["cluster_id"]
        d.rectangle([img.width + 4, a, img.width + 20, max(a, b - 1)],
                    fill=(200, 200, 200) if c < 0 else cols[c])
        d.line([(0, a), (img.width + 20, a)], fill=(0, 40, 200), width=1)
        if b - a >= 12:
            d.text((img.width + 26, (a + b) / 2), "no data" if c < 0 else f"c{c}",
                   fill=(0, 0, 0), font=f, anchor="lm")
    canvas.save(path)


def save_montage(path, descs, labels, margin, per_row=14):
    """One row per cluster: thumbnails of members, LEAST typical first (highest margin), so a
    wrong member is the first thing you see. Red border = thin (assigned by colour/neighbour)."""
    k = max(labels.max() + 1, 1)          # blank intervals (-1) are not shown
    tw, th, pad = 60, 40, 4
    img = Image.new("RGB", (130 + per_row * (tw + pad), k * (th + 2 * pad) + pad), "white")
    d = ImageDraw.Draw(img)
    f = font(12)
    for c in range(k):
        m = [i for i in np.argsort(-margin) if labels[i] == c]
        y = pad + c * (th + 2 * pad)
        d.text((4, y + th / 2), f"c{c}  n={len(m)}", fill=(0, 0, 0), font=f, anchor="lm")
        for j, i in enumerate(m[:per_row]):
            t = descs[i]["thumb"]
            if t.size == 0:
                continue
            x = 130 + j * (tw + pad)
            img.paste(Image.fromarray(t).resize((tw, th), Image.Resampling.NEAREST), (x, y))
            d.rectangle([x - 1, y - 1, x + tw, y + th], outline=(220, 0, 0) if descs[i]["thin"] else (160, 160, 160))
    img.save(path)


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


# ---------------------------------------------------------------- scoring

def read_intervals(path):
    rows = list(csv.DictReader(open(path, newline="")))
    return [dict(top_y=int(r["top_y"]), bottom_y=int(r["bottom_y"]), lithology=r.get("lithology")) for r in rows]


def row_labels(intervals, key, y0, y1):
    out = np.full(y1 - y0, None, dtype=object)
    for r in intervals:
        a, b = max(r["top_y"], y0), min(r["bottom_y"], y1)
        if b > a:
            out[a - y0: b - y0] = r[key]
    return out


BOUNDARY_TOL = 20   # px. example.csv snaps boundaries to the 10 ft depth grid, which sits up to
                    # ~13 px from the visible change, so a tighter tolerance scores the grid, not us.


def score(truth, merged, raw_n):
    """Row ARI: every pixel row gets the predicted cluster and the true lithology, so the score
    doesn't depend on how the column was cut. Boundary P/R: within BOUNDARY_TOL px counts."""
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


# ---------------------------------------------------------------- run

def run(image, bbox, intervals_csv=None, truth_csv=None, y0=None, y1=None, texture="pooled",
        out_dir=None, tag=None, quiet=False):
    box = json.load(open(bbox))["bounding_boxes"]["interpreted_lithology_column"]
    stem = os.path.splitext(os.path.basename(image))[0]
    im = Image.open(image)

    if intervals_csv:
        intervals = read_intervals(intervals_csv)
        source = os.path.basename(intervals_csv)
    else:   # teammate's segmentation on the requested rows of the lithology column
        ya, yb = max(y0 or box[1], box[1]), min(y1 or box[3], box[3])
        crop = im.crop((box[0], ya, box[2], yb))
        intervals = build_intervals(detect_boundaries(crop), ya)
        source = "task1.py segmentation"
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

    out_dir = out_dir or os.path.join(REPO, "clustering", "out")
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


def run_example():
    img = os.path.join(EXAMPLE, EXAMPLE_STEM + ".tif")
    ann = os.path.join(EXAMPLE, EXAMPLE_STEM + ".json")
    truth = os.path.join(EXAMPLE, "example.csv")
    rows = []
    for source, iv in (("true intervals (example.csv)", truth), ("task1.py segmentation", None)):
        for tex in ("pooled", "team"):
            tag = f"{'truth' if iv else 'task1'}_{tex}"
            r = run(img, ann, iv, truth, texture=tex, tag=tag, quiet=True)
            rows.append((source, tex, r))
    print(f"{'input intervals':30s} {'texture':7s} {'clusters':>8s} {'row ARI':>8s} "
          f"{'intervals in -> out':>20s} {'b.recall':>8s} {'b.prec':>7s} {'thin':>5s}")
    for source, tex, r in rows:
        print(f"{source:30s} {tex:7s} {r['n_clusters']:8d} {r['row_ari']:8.3f} "
              f"{r['n_raw']:>9d} -> {r['n_merged']:<8d} {r['b_recall']:8.2f} {r['b_precision']:7.2f} {r['n_thin']:5d}")
    print(f"(truth: {rows[0][2]['n_true']} intervals, 3 lithologies)  outputs in clustering/out/")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image", nargs="?")
    ap.add_argument("bbox", nargs="?")
    ap.add_argument("--intervals", help="CSV with top_y,bottom_y; default: run task1.py segmentation")
    ap.add_argument("--truth", help="CSV with top_y,bottom_y,lithology for scoring")
    ap.add_argument("--y0", type=int)
    ap.add_argument("--y1", type=int)
    ap.add_argument("--texture", choices=["pooled", "team"], default="pooled")
    ap.add_argument("--out")
    ap.add_argument("--example", action="store_true", help="run both checks on the example")
    a = ap.parse_args()
    if a.example:
        return run_example()
    if not (a.image and a.bbox):
        ap.error("give IMAGE and BBOX_JSON, or --example")
    run(a.image, a.bbox, a.intervals, a.truth, a.y0, a.y1, a.texture, a.out)


if __name__ == "__main__":
    main()
