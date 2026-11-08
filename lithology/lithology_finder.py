#!/usr/bin/env python3
"""
Locate the interpreted-lithology column in a full well-log page (computer vision only, no OCR).

Idea: every track on a log page is framed by two long, thick vertical lines. The lithology
column is the framed track that is filled with a repeated hatch-pattern texture (short horizontal
dashes, dots, bars...) rather than curves, grid lines or numbers. We
  1. shrink the (huge, 1-bit) page while keeping thick lines,
  2. find long vertical lines and TRACE them down the page (scans drift sideways, so lines are
     followed block by block instead of assumed straight),
  3. pair neighbouring lines into candidate tracks of plausible width,
  4. score each candidate by what is inside it (pattern texture, not curves/grid/numbers),
  5. return the best one as a box [x0, y0, x1, y1] covering the whole drifting column.

Usage:
  python lithology_finder.py log.tif [--debug debug.png] [--annotation ann.json]
From code:
  from lithology_finder import find_lithology_column
  res = find_lithology_column(path)       # None if nothing column-like was found
  res["bbox"], res["confidence"]
"""
import argparse
import json
import sys

import cv2
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
BLOCK = 600          # rows (at working scale) per tracing block


# ----------------------------------------------------------------------------
# loading: huge 1-bit TIFFs are normal for logs, so shrink in strips
# ----------------------------------------------------------------------------
def load_ink_small(path, factor=4, strip=4000):
    """Return (ink, W, H): ink in [0,1] (1 = black) at 1/factor scale (box average, so thick lines
    survive and speckle fades), plus the original page size."""
    im = Image.open(str(path))
    W, H = im.size
    parts = []
    for y in range(0, H, strip):
        s = im.crop((0, y, W, min(H, y + strip))).convert("L").reduce(factor)
        parts.append(255 - np.asarray(s))
    return np.vstack(parts).astype(np.float32) / 255.0, W, H


# ----------------------------------------------------------------------------
# step 1-2: long vertical lines, traced down the page
# ----------------------------------------------------------------------------
def trace_lines(ink, f, min_len_frac=0.4, link=4):
    """Return tracks: dicts with y0,y1 (block index range) and centres/half-widths per block, in
    ORIGINAL pixels. A track is a long, thick, near-vertical line that may drift sideways."""
    h, _ = ink.shape
    mask = (ink > 0.3).astype(np.uint8)
    k = max(50, int(0.006 * h))                      # long-line filter: ~0.6% of page height
    vlines = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, k)))
    nb = (h + BLOCK - 1) // BLOCK
    per_block = []
    for b in range(nb):
        p = vlines[b * BLOCK:(b + 1) * BLOCK].mean(0)
        xs = np.where(p > 0.7)[0]
        groups = np.split(xs, np.where(np.diff(xs) > 2)[0] + 1) if len(xs) else []
        per_block.append([((g[0] + g[-1] + 1) / 2.0, (g[-1] - g[0] + 1) / 2.0) for g in groups])
    tracks, open_t = [], []
    for b, lines in enumerate(per_block):
        nxt = []
        for c, hw in lines:
            best = min(open_t, key=lambda t: abs(t["c"][-1] - c), default=None)
            if best is not None and abs(best["c"][-1] - c) <= link and best not in nxt:
                best["c"].append(c); best["hw"].append(hw); best["last"] = b
                nxt.append(best)
            else:
                t = dict(first=b, last=b, c=[c], hw=[hw])
                tracks.append(t); nxt.append(t)
        open_t = nxt
    out = []
    for t in tracks:
        if t["last"] - t["first"] + 1 >= min_len_frac * nb:
            out.append(dict(first=t["first"], last=t["last"],
                            c=np.array(t["c"]) * f, hw=np.array(t["hw"]) * f))
    out.sort(key=lambda t: t["c"].mean())
    return out, vlines


# ----------------------------------------------------------------------------
# step 3-4: candidate tracks and what is inside them
# ----------------------------------------------------------------------------
def _interior(ink, f, left, right):
    """Stack the inside of a (possibly drifting) track into a straight, fixed-width array."""
    b0, b1 = max(left["first"], right["first"]), min(left["last"], right["last"])
    if b1 < b0:                                    # the two lines never co-exist: not a track
        return None, b0, b1
    w = int(min((right["c"][b - right["first"]] - left["c"][b - left["first"]]) for b in range(b0, b1 + 1)) / f) - 10
    if w < 8:
        return None, b0, b1
    rows = []
    for b in range(b0, b1 + 1):
        x0 = int(left["c"][b - left["first"]] / f) + 5
        rows.append(ink[b * BLOCK:(b + 1) * BLOCK, x0:x0 + w])
    return np.vstack(rows), b0, b1


def texture_features(inner):
    """Features of the inside of a track, in working-scale pixels."""
    m = (inner > 0.3).astype(np.uint8)
    n = max(1, int(m.sum()))
    rows = inner.mean(1)
    # horizontal strokes (dashes): ink surviving a short horizontal opening
    hs = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1)))
    # long vertical strokes (grid lines, curves' verticals)
    vs = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 60)))
    # periodicity of the row profile in the densest stretch
    j = int(np.argmax(np.convolve(rows, np.ones(1500) / 1500, "valid"))) if len(rows) > 1500 else 0
    seg = rows[j:j + 1500] - rows[j:j + 1500].mean()
    ac = np.correlate(seg, seg, "full")[len(seg) - 1:]
    ac = ac / (ac[0] + 1e-9)
    return dict(mean_ink=float(inner.mean()),
                content_rows=float(np.mean(np.convolve(rows, np.ones(100) / 100, "same") > 0.03)),
                hstroke=float(hs.sum() / n), vstroke=float(vs.sum() / n),
                periodic=float(ac[3:80].max()) if len(ac) > 80 else 0.0)


def candidates(ink, f, W, tracks):
    out = []
    for a, b in zip(tracks[:-1], tracks[1:]):
        gap = float(np.mean(b["c"][:len(b["c"])]) - np.mean(a["c"]))
        if not 0.012 * W <= gap <= 0.12 * W:
            continue
        inner, b0, b1 = _interior(ink, f, a, b)
        if inner is None:
            continue
        out.append(dict(left=a, right=b, width=gap, rows=(b0, b1), feats=texture_features(inner)))
    return out


def score_candidate(c, W):
    """0..1: how much does the inside of this track look like a lithology (pattern) column?"""
    f = c["feats"]
    width = c["width"] / W
    width_prior = 1.0 if 0.03 <= width <= 0.075 else max(0.0, 1 - abs(width - (0.03 if width < 0.03 else 0.075)) / 0.03)
    no_grid = 1.0 - min(1.0, f["vstroke"] / 0.05)           # grid lines / curve verticals => not a pattern fill
    filled = min(1.0, f["content_rows"] / 0.5)              # pattern fills most of the page height
    textured = min(1.0, f["hstroke"] / 0.15) if f["mean_ink"] > 0.02 else 0.0
    return float(max(0.0, f["periodic"]) * no_grid * filled * textured * width_prior)


def _line_extent(track, vlines, f):
    """Fallback: first/last row (original px) where the traced vertical line exists (+/- one block)."""
    x0 = int(max(0, (track["c"] - track["hw"]).min() / f - 1))
    x1 = int((track["c"] + track["hw"]).max() / f + 2)
    ys = np.where(vlines[:, x0:x1].any(axis=1))[0]
    lo, hi = (track["first"] - 1) * BLOCK, (track["last"] + 2) * BLOCK
    ys = ys[(ys >= lo) & (ys < hi)]
    return int(ys[0] * f), int((ys[-1] + 1) * f)


def _clamped(track, b):
    i = min(max(b - track["first"], 0), len(track["c"]) - 1)
    return track["c"][i]


def body_extent(path, ink, f, L, R, vlines):
    """The column body is the longest stretch between thick horizontal rules that span the whole
    track (the frame lines also run through the repeated header/footer blocks, so they alone
    would overshoot). Returns (y0, y1) in original px."""
    nb = ink.shape[0] // BLOCK + 1
    b0, b1 = max(min(L["first"], R["first"]) - 1, 0), min(max(L["last"], R["last"]) + 1, nb - 1)
    cov = []
    for b in range(b0, b1 + 1):
        xl, xr = int(_clamped(L, b) / f) + 5, int(_clamped(R, b) / f) - 5
        cov.append((ink[b * BLOCK:(b + 1) * BLOCK, xl:xr] > 0.3).mean(1))
    cov = np.concatenate(cov) >= 0.95                      # full-width ink = a ruled line
    rows = np.where(cov)[0]
    if len(rows) == 0:
        return _line_extent(L, vlines, f)
    groups = np.split(rows, np.where(np.diff(rows) > 2)[0] + 1)
    rules = [((g[0] + b0 * BLOCK) * f, (g[-1] + 1 + b0 * BLOCK) * f) for g in groups]   # original px
    ends = _line_extent(L, vlines, f)
    rules = [(ends[0], ends[0])] + rules + [(ends[1], ends[1])]
    gaps = [(rules[i + 1][0] - rules[i][1], i) for i in range(len(rules) - 1)]
    _, i = max(gaps)
    y0, y1 = rules[i][0], rules[i + 1][1]
    if rules[i][0] == rules[i][1]:                         # no real rule on that side
        y0 = rules[i][0]
    return int(y0), int(y1)


def refine_rule_edge(path, x0, x1, y, want, win=16):
    """Pin a rule edge to the pixel: `want` is 'top' or 'bottom'. Looks at full resolution."""
    im = Image.open(str(path))
    y0 = max(0, y - win); y1 = min(im.size[1], y + win)
    a = np.asarray(im.crop((x0, y0, x1, y1)).convert("L")) < 128
    full = np.where(a.mean(1) >= 0.95)[0]
    if len(full) == 0:
        return y
    return int(y0 + (full[0] if want == "top" else full[-1] + 1))


# ----------------------------------------------------------------------------
# main entry
# ----------------------------------------------------------------------------
def find_by_lines(path, factor=4, tie_frac=0.6, return_debug=False):
    """Locate the interpreted-lithology column. Returns dict(bbox=[x0,y0,x1,y1] in original px,
    confidence, score, alternatives) or None."""
    ink, W, H = load_ink_small(path, factor)
    tracks, vlines = trace_lines(ink, factor)
    cands = candidates(ink, factor, W, tracks)
    for c in cands:
        c["score"] = score_candidate(c, W)
    debug = dict(ink=ink, factor=factor, tracks=tracks, candidates=cands)
    if not cands or max(c["score"] for c in cands) < 0.2:
        return (None, debug) if return_debug else None

    best = max(c["score"] for c in cands)
    similar = [c for c in cands if c["score"] >= tie_frac * best]
    # Logs usually carry a "cuttings" lithology column beside the depth column AND the interpreted
    # one near the right, so among equally pattern-like tracks take the rightmost.
    pick = max(similar, key=lambda c: c["left"]["c"].mean())

    L, R = pick["left"], pick["right"]
    x0 = int((L["c"] - L["hw"]).min())
    x1 = int(np.ceil((R["c"] + R["hw"]).max()))
    y0, y1 = body_extent(path, ink, factor, L, R, vlines)
    xi0, xi1 = int(np.median(L["c"])) + 12, int(np.median(R["c"])) - 12
    y0 = refine_rule_edge(path, xi0, xi1, y0, "top")        # note: x range follows the median drift,
    y1 = refine_rule_edge(path, xi0, xi1, y1, "bottom")     # rules are far wider than the drift

    def box(c):
        a, b = c["left"], c["right"]
        ya, yb = body_extent(path, ink, factor, a, b, vlines)
        return [int((a["c"] - a["hw"]).min()), ya, int(np.ceil((b["c"] + b["hw"]).max())), yb]
    others = [dict(bbox=box(c), score=round(c["score"], 3)) for c in similar if c is not pick]
    conf = float(min(1.0, pick["score"]) * (0.6 if others else 1.0))
    res = dict(bbox=[x0, y0, x1, y1], score=float(pick["score"]), confidence=conf,
               alternatives=others, page=(W, H))
    debug["pick"] = pick
    return (res, debug) if return_debug else res



# ----------------------------------------------------------------------------
# mode B: digital / greyscale mudlogs (narrow columns, grey fills, often no thick frame lines)
# ----------------------------------------------------------------------------
def load_gray(path, max_w=2400):
    im = Image.open(str(path)).convert("L")
    f = max(1, int(round(im.width / max_w)))
    if f > 1:
        im = im.reduce(f)
    return np.asarray(im), f


def fill_profile(gray, win=300, white=225):
    """For every x: fraction of rows that are non-white, taken per window of `win` rows, then the
    MEDIAN over windows (robust to headers, footers and blank stretches)."""
    nonwhite = gray < white
    H = nonwhite.shape[0]
    wins = np.array([nonwhite[i:i + win].mean(0) for i in range(0, max(1, H - win + 1), win)])
    return np.median(wins, 0), wins


def band_candidates(gray, min_frac=0.015, max_frac=0.065, level=0.4):
    """Narrow, sharply bounded vertical bands that are filled (pattern / grey) nearly everywhere."""
    W = gray.shape[1]
    med, wins = fill_profile(gray)
    sm = np.convolve(med, np.ones(3) / 3, "same")
    on = sm >= level
    xs = np.where(on)[0]
    if len(xs) == 0:
        return [], med
    runs = np.split(xs, np.where(np.diff(xs) > 3)[0] + 1)          # merge gaps of up to 3 px
    out = []
    for r in runs:
        a, b = int(r[0]), int(r[-1]) + 1
        if not min_frac * W <= b - a <= max_frac * W:
            continue
        # pin each edge to a sharp step of the raw profile near the thresholded edge: among the steps
        # that are nearly as steep as the steepest, take the one closest to that edge (a thin grid
        # line a few px outside the band is a steep step too, but it is not the band's edge)
        ga = np.arange(max(1, a - 8), min(W - 1, a + 8))
        if len(ga):
            st = med[ga + 1] - med[ga - 1]
            ok = np.where(st >= 0.8 * st.max())[0]
            a = int(ga[ok[np.argmin(np.abs(ga[ok] - a))]]) + 1
        gb = np.arange(max(1, b - 8), min(W - 1, b + 8))
        if len(gb):
            st = med[gb - 1] - med[gb + 1]
            ok = np.where(st >= 0.8 * st.max())[0]
            b = int(gb[ok[np.argmin(np.abs(gb[ok] - b))]])
        inside = wins[:, a:b].mean(1)
        continuity = float(np.mean(inside >= 0.25))                  # fraction of the page it is filled
        lo, hi = max(0, a - 6), min(W, b + 6)
        edge = float(sm[a:b].mean() - max(sm[lo:a].mean() if a > lo else 0, sm[b:hi].mean() if hi > b else 0))
        out.append(dict(x0=a, x1=b, continuity=continuity, edge=edge, level=float(sm[a:b].mean())))
    return out, med


def band_extent(gray, x0, x1, level, win=50):
    """Vertical extent of a band, deliberately coarse (a few dozen rows on a page tens of thousands
    of rows tall costs almost nothing): windows where either side border or the fill itself is
    present, longest run with big gaps tolerated, then snapped to a rule only if one is very close."""
    H, W = gray.shape
    def presence(a, b):
        return (gray[:, max(0, a):min(W, b)] < 200).any(1)
    sig = presence(x0 - 3, x0 + 3) | presence(x1 - 3, x1 + 3)
    n = H // win
    on = sig[:n * win].reshape(n, win).mean(1) >= 0.9
    ws = np.where(on)[0]
    if len(ws) == 0:
        return 0, H
    runs = np.split(ws, np.where(np.diff(ws) > 6)[0] + 1)          # gaps up to 6 windows (300 rows)
    r = max(runs, key=len)
    y0, y1 = int(r[0] * win), int((r[-1] + 1) * win)
    wide = np.where((gray < 200).mean(1) > 0.8)[0]                  # page-wide dark rules
    groups = np.split(wide, np.where(np.diff(wide) > 2)[0] + 1) if len(wide) else []
    fwd = int(0.4 * W)        # the data zone starts a little AFTER the first border activity (the
    #                           headers above it have rules and borders of their own), and ends a
    #                           little BEFORE the last one
    def snap_top(y):
        near = [g for g in groups if y - 60 <= g[0] <= y + fwd]
        return int(near[-1][0]) if near else y
    def snap_bottom(y):
        near = [g for g in groups if y - fwd <= g[-1] <= y + 60]
        return int(near[0][-1] + 1) if near else y
    return snap_top(y0), snap_bottom(y1)


def find_by_bands(path, tie_continuity=0.45, return_debug=False):
    gray, f = load_gray(path)
    cands, med = band_candidates(gray)
    good = [c for c in cands if c["continuity"] >= tie_continuity and c["edge"] >= 0.08]
    debug = dict(mode="bands", candidates=cands, med=med, factor=f)
    if not good:
        return (None, debug) if return_debug else None
    pick = max(good, key=lambda c: c["x0"])          # rightmost: the interpreted column sits last
    y0, y1 = band_extent(gray, pick["x0"], pick["x1"], pick["level"])
    res = dict(bbox=[pick["x0"] * f, y0 * f, pick["x1"] * f, y1 * f], score=pick["continuity"],
               confidence=float(min(1.0, pick["continuity"]) * min(1.0, pick["edge"] / 0.4) * (0.7 if len(good) > 1 else 1.0)),
               alternatives=[dict(bbox=[c["x0"] * f, 0, c["x1"] * f, gray.shape[0] * f]) for c in good if c is not pick],
               mode="bands")
    return (res, debug) if return_debug else res



# ----------------------------------------------------------------------------
# mode C: framed narrow track (works when the column is blank / unfilled on most of the page)
# ----------------------------------------------------------------------------
def load_ink_vreduced(path, vfactor=8, strip=8000):
    """Ink in [0,1] with full horizontal resolution (thin lines stay visible) but rows reduced."""
    im = Image.open(str(path))
    W, H = im.size
    parts = []
    for y in range(0, H, strip):
        s = im.crop((0, y, W, min(H, y + strip))).convert("L")
        s = s.resize((W, max(1, s.height // vfactor)), Image.BOX)
        parts.append(255 - np.asarray(s))
    return np.vstack(parts).astype(np.float32) / 255.0, W, H


def find_by_frames(path, min_frac=0.015, max_frac=0.065, min_thick=1.2, return_debug=False):
    """Pairs of neighbouring THICK vertical lines (page grids are made of thin ones) that are a
    plausible column width apart; the rightmost pair is the interpreted-lithology column."""
    ink, W, H = load_ink_vreduced(path)
    vf = H / ink.shape[0]
    tracks, vlines = trace_lines(ink, 1, min_len_frac=0.3)
    thick = [t for t in tracks if float(np.median(t["hw"])) >= min_thick]
    pairs = []
    for a, b in zip(thick[:-1], thick[1:]):
        gap = float(np.median(b["c"]) - np.median(a["c"]))
        if min_frac * W <= gap <= max_frac * W:
            pairs.append((a, b))
    debug = dict(mode="frames", n_tracks=len(tracks), n_thick=len(thick), n_pairs=len(pairs))
    if not pairs:
        return (None, debug) if return_debug else None
    a, b = pairs[-1]
    x0 = int((a["c"] - a["hw"]).min()); x1 = int(np.ceil((b["c"] + b["hw"]).max()))
    gray, f = load_gray(path)
    y0, y1 = band_extent(gray, x0, x1, 0.3)
    res = dict(bbox=[x0, y0, x1, y1], score=0.5, confidence=0.4 * (0.7 if len(pairs) > 1 else 1.0),
               alternatives=[dict(bbox=[int((p["c"] - p["hw"]).min()), 0, int(np.ceil((q["c"] + q["hw"]).max())), H])
                             for p, q in pairs[:-1]], mode="frames")
    return (res, debug) if return_debug else res


def find_lithology_column(path, return_debug=False, scan_width=3000):
    """Locate the interpreted-lithology column. Returns dict(bbox=[x0,y0,x1,y1], confidence, mode,
    alternatives) or None.

    Wide pages (>= scan_width px) are scans with thick frame lines: trace those first. Narrower pages
    are digital mudlogs: look for a narrow filled band first. Either way the other methods are the
    fallbacks, ending with a framed-but-blank track."""
    W = Image.open(str(path)).size[0]
    def lines(p, rd):
        out = find_by_lines(p, return_debug=True)
        if out[0] is not None:
            out[0]["mode"] = "lines"
        return out if rd else out[0]
    methods = [lines, find_by_bands, find_by_frames] if W >= scan_width else [find_by_bands, find_by_frames]
    out = (None, {})
    for m in methods:
        out = m(path, return_debug=True) if m is not lines else lines(path, True)
        if out[0] is not None:
            break
    return out if return_debug else out[0]


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def draw_debug(debug, res, out_png, shrink=0.25):
    """Page overview: blue = candidate tracks (with score), red = chosen column."""
    ink, f = debug["ink"], debug["factor"]
    small = cv2.resize(1.0 - ink, None, fx=shrink, fy=shrink, interpolation=cv2.INTER_AREA)
    vis = cv2.cvtColor((small * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    s = shrink / f                                   # original px -> overview px
    for c in debug["candidates"]:
        x0, x1 = int(c["left"]["c"].mean() * s), int(c["right"]["c"].mean() * s)
        cv2.rectangle(vis, (x0, 0), (x1, vis.shape[0] - 1), (255, 120, 0), 1)
        cv2.putText(vis, f"{c['score']:.2f}", (x0 + 2, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
    if res:
        x0, y0, x1, y1 = res["bbox"]
        cv2.rectangle(vis, (int(x0 * s), int(y0 * s)), (int(x1 * s), int(y1 * s)), (0, 0, 255), 2)
    cv2.imwrite(out_png, vis)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image")
    ap.add_argument("--debug", help="write a debug overview PNG")
    ap.add_argument("--annotation", help="annotation json; prints IoU against its lithology box")
    a = ap.parse_args()
    res, dbg = find_lithology_column(a.image, return_debug=True)
    if a.debug:
        draw_debug(dbg, res, a.debug)
    if res is None:
        sys.exit("no lithology-column-like track found")
    print(json.dumps({k: v for k, v in res.items() if k != "page"}))
    if a.annotation:
        gt = json.load(open(a.annotation))["bounding_boxes"]["interpreted_lithology_column"]
        print("annotation:", gt, "IoU = %.3f" % iou(res["bbox"], gt))


if __name__ == "__main__":
    main()