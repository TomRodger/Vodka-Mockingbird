#!/usr/bin/env python3
"""
Cut a lithology legend image into individual pattern swatches.

Two strategies, chosen automatically:

  A) "label-anchored" (swatch to the LEFT of its label, no frames) -- e.g. big
     multi-column legends. OCR finds the labels; each label defines a row; the
     swatch is the ink in the strip left of the label. Handles sparse patterns
     (dots, crosses) and blank swatches (e.g. CLAY) because the position comes
     from the label, not from the ink.

  B) "component" (framed swatches, label BELOW) -- e.g. small boxed legends.
     Text is painted out, remaining ink is grouped by connected components,
     and each box is paired with the nearest label.

Usage:
  python legend_cutter.py legend.png out_dir [--layout auto|left|boxes] [--no-ocr]

Outputs in out_dir: one PNG per swatch, _index.csv, _debug.png (overlay).
"""
import argparse
import csv
import re
import sys
from pathlib import Path

import cv2
import numpy as np

try:
    import os
    import pytesseract
    # Windows default install location (no-op elsewhere)
    _win_tess = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if os.path.exists(_win_tess):
        pytesseract.pytesseract.tesseract_cmd = _win_tess
except ImportError:
    pytesseract = None


# ----------------------------------------------------------------------------
# OCR
# ----------------------------------------------------------------------------
def ocr_labels(img, scale=None, min_conf=40):
    """Text *lines* as dicts(text,x,y,w,h) in original pixel coords."""
    H, W = img.shape[:2]
    if scale is None:   # small images are upscaled 3x; big ones less (cap ~3000 px)
        scale = 3.0 if max(H, W) <= 1500 else max(1.0, 3000.0 / max(H, W))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    big = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    d = pytesseract.image_to_data(big, config="--psm 11", output_type=pytesseract.Output.DICT)
    words = []
    for i, t in enumerate(d["text"]):
        t = t.strip()
        if not t or float(d["conf"][i]) < min_conf:
            continue
        x, y, w, h = (d[k][i] / scale for k in ("left", "top", "width", "height"))
        words.append(dict(text=t, x=x, y=y, w=w, h=h, cy=y + h / 2, conf=float(d["conf"][i])))
    if not words:
        return []

    # reference text height from words that look like real words, then discard
    # tokens whose height is far off it (junk read from hatch patterns is tall or tiny)
    real = [w["h"] for w in words if w["conf"] >= 60 and sum(c.isalpha() for c in w["text"]) >= 4]
    ref = float(np.median(real if len(real) >= 3 else [w["h"] for w in words]))
    words = [w for w in words if w["text"] in ("-", "–", "—") or 0.5 * ref <= w["h"] <= 1.8 * ref]
    if not words:
        return []
    med_h = ref

    # 1) group words into text lines by vertical centre
    words.sort(key=lambda w: w["cy"])
    rows, cur = [], [words[0]]
    for wd in words[1:]:
        if abs(wd["cy"] - np.mean([c["cy"] for c in cur])) < 0.5 * med_h:
            cur.append(wd)
        else:
            rows.append(cur); cur = [wd]
    rows.append(cur)

    # 2) within a row, chain words left->right while the gap is small
    lines = []
    for row in rows:
        row.sort(key=lambda w: w["x"])
        chain = [row[0]]
        for wd in row[1:]:
            prev = chain[-1]
            if wd["x"] - (prev["x"] + prev["w"]) < 0.9 * med_h:
                chain.append(wd)
            else:
                lines.append(chain); chain = [wd]
        lines.append(chain)

    # 3) a real label contains at least one real word (>=4 letters)
    out = []
    for ch in lines:
        if not any(sum(c.isalpha() for c in w["text"]) >= 4 for w in ch):
            continue
        while len(ch) > 1 and sum(c.isalpha() for c in ch[0]["text"]) < 4 and ch[0]["text"] not in ("-",):
            if len(ch[0]["text"]) <= 2 and ch[0]["x"] + ch[0]["w"] < ch[1]["x"] - 0.5 * med_h:
                ch = ch[1:]
            else:
                break
        x1 = min(w["x"] for w in ch); y1 = min(w["y"] for w in ch)
        x2 = max(w["x"] + w["w"] for w in ch); y2 = max(w["y"] + w["h"] for w in ch)
        out.append(dict(text=" ".join(w["text"] for w in ch), x=x1, y=y1, w=x2 - x1, h=y2 - y1))
    return out


def text_unit(labels, default=10.0):
    """Typical text height in px: every pixel threshold below scales with this."""
    return float(np.median([l["h"] for l in labels])) if labels else default


def left_evidence(img, labels):
    """Fraction of labels with ink right next to them on the LEFT -> swatch-left layout."""
    if not labels:
        return 0.0
    mask = ink_mask(img); u = text_unit(labels); hit = 0
    for l in labels:
        bb = ink_bbox(mask, l["x"] - 2.5 * u, l["y"], l["x"] - 1, l["y"] + l["h"], min_px=max(6, int(0.5 * u)))
        hit += bb is not None
    return hit / len(labels)


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def ink_mask(img, white_thresh=235):
    """Anything not near-white: dark linework OR coloured fill."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return (((gray < white_thresh) | ((hsv[..., 1] > 40) & (hsv[..., 2] > 60)))
            .astype(np.uint8) * 255)


def ink_bbox(mask, x0, y0, x1, y1, min_px=6):
    """Tight bbox of ink inside a window, or None."""
    x0, y0 = max(int(x0), 0), max(int(y0), 0)
    x1, y1 = min(int(x1), mask.shape[1]), min(int(y1), mask.shape[0])
    if x1 <= x0 or y1 <= y0:
        return None
    ys, xs = np.where(mask[y0:y1, x0:x1] > 0)
    if len(xs) < min_px:
        return None
    return (x0 + xs.min(), y0 + ys.min(), xs.max() - xs.min() + 1, ys.max() - ys.min() + 1)


def cluster_columns(labels, tol=20, min_frac=0.5):
    """Group labels into columns by left edge."""
    cols = []
    for lb in sorted(labels, key=lambda l: l["x"]):
        for c in cols:
            if abs(c["x"] - lb["x"]) < tol:
                c["items"].append(lb)
                c["x"] = np.mean([i["x"] for i in c["items"]])
                break
        else:
            cols.append(dict(x=lb["x"], items=[lb]))
    # real label columns are far fuller than clusters of OCR junk read off the swatches
    if cols:
        biggest = max(len(c["items"]) for c in cols)
        cols = [c for c in cols if len(c["items"]) >= max(3, min_frac * biggest)]
    for c in cols:
        c["items"].sort(key=lambda l: l["y"])
    return cols


# ----------------------------------------------------------------------------
# Strategy A: label-anchored, swatch left of label
# ----------------------------------------------------------------------------
def strategy_left(img, labels, max_swatch_w=None):
    """Returns (list of (bbox, label_text), ink_fraction)."""
    if max_swatch_w is None:
        max_swatch_w = 7 * text_unit(labels)
    mask = ink_mask(img)
    H, W = mask.shape
    cols = cluster_columns(labels)
    if len(cols) == 0:
        return [], 0.0
    labels = [l for c in cols for l in c["items"]]

    found = []   # (label, bbox or None)
    for ci, c in enumerate(cols):
        items = c["items"]
        ys = np.array([l["y"] + l["h"] / 2 for l in items])
        pitch = np.median(np.diff(ys)) if len(ys) > 1 else 25.0
        # left bound: right edge of the previous column's labels
        left = 0 if ci == 0 else max(l["x"] + l["w"] for l in cols[ci - 1]["items"]) + 2
        for l in items:
            cy = l["y"] + l["h"] / 2
            x0 = max(left, l["x"] - max_swatch_w)
            bb = ink_bbox(mask, x0, cy - pitch / 2 + 1, l["x"] - 2, cy + pitch / 2 - 1)
            found.append((l, bb))

        # fill rows OCR missed (gap bigger than ~1.6 pitch) as label-less rows
        for a, b in zip(items[:-1], items[1:]):
            gap = (b["y"] + b["h"] / 2) - (a["y"] + a["h"] / 2)
            n_missing = int(round(gap / pitch)) - 1
            for k in range(1, n_missing + 1):
                cy = a["y"] + a["h"] / 2 + k * gap / (n_missing + 1)
                fake = dict(text="", x=c["x"], y=cy - a["h"] / 2, w=a["w"], h=a["h"])
                x0 = max(left, c["x"] - max_swatch_w)
                found.append((fake, ink_bbox(mask, x0, cy - pitch / 2 + 1, c["x"] - 2, cy + pitch / 2 - 1)))

    boxes = [bb for _, bb in found if bb is not None]
    if not boxes:
        return [], 0.0
    ink_frac = len(boxes) / len(found)

    # uniform crop size so every swatch looks alike; centre blank ones on the column's swatch x
    ws = np.array([b[2] for b in boxes]); hs = np.array([b[3] for b in boxes])
    W_ = int(np.percentile(ws, 85)); H_ = int(np.percentile(hs, 85))
    pitches = [np.median(np.diff([l["y"] for l in c["items"]])) for c in cols if len(c["items"]) > 1]
    if pitches:
        H_ = min(H_, int(np.median(pitches)) - 4)   # never reach into the neighbouring row
    med_cx = {}
    for ci, c in enumerate(cols):
        cxs = [b[0] + b[2] / 2 for (l, b) in found if b is not None and abs(l["x"] - c["x"]) < 20]
        med_cx[ci] = np.median(cxs) if cxs else None

    out = []
    for l, bb in found:
        ci = min(range(len(cols)), key=lambda i: abs(cols[i]["x"] - l["x"]))
        cy = l["y"] + l["h"] / 2
        if bb is not None:
            cx = bb[0] + bb[2] / 2
            cy = bb[1] + bb[3] / 2
        elif med_cx[ci] is not None:
            cx = med_cx[ci]
        else:
            continue
        out.append(((int(cx - W_ / 2), int(cy - H_ / 2), W_, H_), l["text"]))
    return sorted(out, key=lambda o: (o[0][0] // 50, o[0][1])), ink_frac


# ----------------------------------------------------------------------------
# Strategy B: connected components, framed swatches, label below/right
# ----------------------------------------------------------------------------
def side_coverage(mask, x, y, w, h):
    """Fraction of each side (top, bottom, left, right) of a box that carries ink."""
    x, y, w, h = int(x), int(y), int(w), int(h)
    t = max(2, int(0.08 * min(w, h)))
    sub = mask[y:y + h, x:x + w] > 0
    return (sub[:t].any(axis=0).mean(), sub[-t:].any(axis=0).mean(),
            sub[:, :t].any(axis=1).mean(), sub[:, -t:].any(axis=1).mean())


def strategy_boxes(img, labels, dilate=None, min_w=None):
    """Framed swatches: find rectangles by their outline, then name them from nearby labels.

    OCR is NOT used to find swatches (its boxes can clip frames and its junk tokens
    mislead); a frame is any blob with ink along >=3 of its 4 sides -- text blobs
    don't have that.
    """
    u = text_unit(labels)
    dilate = dilate or max(3, int(0.25 * u) | 1)   # small: must not bridge the gap to the label
    min_w = min_w or 2 * u
    mask = ink_mask(img)
    glued = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (dilate, dilate)))
    n, _, st, _ = cv2.connectedComponentsWithStats(glued, connectivity=8)
    H, W = mask.shape
    boxes = []
    for i in range(1, n):
        x, y, w, h, _a = st[i]
        if w < min_w or h < u or w * h > 0.5 * W * H:
            continue
        bb = ink_bbox(mask, x, y, x + w, y + h)
        if not bb:
            continue
        sides = sorted(side_coverage(mask, *bb), reverse=True)
        if bb[3] < 1.2 * u:                                       # shorter than a line of text
            continue
        full = sides[2] >= 0.75                                   # frame with >=3 sides
        partial = sides[1] >= 0.8                                 # frame cut off on one end
        if full or partial:
            boxes.append(bb)
    if len(boxes) >= 4:  # drop size outliers (titles, merged blobs)
        mw = np.median([b[2] for b in boxes]); mh = np.median([b[3] for b in boxes])
        # height bound is loose on purpose: frames clipped by the image edge are short but still full width
        boxes = [b for b in boxes if 0.5 * mw <= b[2] <= 1.8 * mw and 0.3 * mh <= b[3] <= 1.8 * mh]

    used, out = set(), []
    for b in sorted(boxes, key=lambda b: (int(b[1] // u), b[0])):
        bx, by, bw, bh = b
        cx, cy = bx + bw / 2, by + bh / 2
        best, bd = None, 1e9
        for j, l in enumerate(labels):
            if j in used:
                continue
            lcx, lcy = l["x"] + l["w"] / 2, l["y"] + l["h"] / 2
            right = l["x"] >= bx + bw * 0.5 and abs(lcy - cy) < bh * 0.8
            below = l["y"] >= by + bh * 0.5 and abs(lcx - cx) < max(bw, l["w"]) * 0.9
            d = (l["x"] - (bx + bw)) if right else (l["y"] - (by + bh)) if below else None
            if d is not None and d < bd:
                best, bd = j, d
        if best is not None:
            used.add(best)
        out.append((b, labels[best]["text"] if best is not None else ""))
    return out


# ----------------------------------------------------------------------------
def slug(s, fallback):
    return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_").lower() or fallback


# ----------------------------------------------------------------------------
# Naming: per-swatch OCR + snapping to known lithology names
# ----------------------------------------------------------------------------
KNOWN_NAMES = "ALTERED ZONE|ANDESITE|ANHYDRITE|BASALT|BENTONITE|BIOTITIZATION|BRECCIA|CALCARENITE|CALCAREOUS TUFF|CALCILUTITE|CARBONATES|CARBONACEOUS MAT|CARBONACEOUS SH|CEMENT CONTAM.|CHALK|CRYSTALLINE TUFF|CHERT - ARGILL|CHERT - GLASSY|CHERT - PORCEL|CHERT - TIGER STRIPE|CHERT - UNDIFF|CLAY|CLAY-MUDSTONE|CLYST-TUFFACEOUS|CHLORITIZATION|COAL|CONGLOMERATE|CONGL. SAND|CONGL. SANDSTONE|COQUINA|DACITE|DIATOMITE|DIORITE|DOLOSTONE|FELSIC SILIC DIKE|FOSSIL|GABBRO|GLASSY TUFF|GRANITE|GRANITE WASH|GRANODIORITE|GYPSUM|HALITE|HORNBL-QTZ-DIO|IGNEOUS (ACIDIC)|IGNEOUS (BASIC)|INTRUSIVES|KAOLINITIC|LIMESTONE|LITHIC TUFF|MARL - DOLO|MARL - CALC|METAMORPHICS|MUDSTONE|OBSIDIAN|PALEOSOL|PHOSPHATE|PORCELANITE|PORCELANEOUS CLYST|PYRITE|PYROCLASTICS|QUARTZ DIORITE|QUARTZ LATITE|QUARTZ MONZONITE|RECRYSTALLIZED CALCITE|RHYOLITE|SALT|SAND|SANDSTONE|SANDSTONE-TUFFACEOUS|SERICITIZATION|SERPENTINE|SHALE|SHALE TUFFACEOUS|SHELL FRAGMENTS|SIDERITE|SILICIFICATION|SILTSTONE|SILTST-TUFFACEOUS|TUFF|VOLCANICLASTICS SEDS|VOLCANICS|DOLOMITE|SILTY SANDSTONE|SAND AND SANDSTONE".split("|")


def _norm(t):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", t.lower())).strip()


def load_names(path=None):
    names = list(KNOWN_NAMES)
    if path:
        names += [l.strip() for l in open(path, encoding="utf-8") if l.strip()]
    return names


def snap_name(text, names, cutoff=0.8):
    """Replace OCR text by the closest known lithology name (if close enough)."""
    import difflib
    n = _norm(text)
    if not n:
        return ""
    table = {_norm(k): k for k in names}
    hit = difflib.get_close_matches(n, list(table), n=1, cutoff=cutoff)
    return table[hit[0]].title() if hit else text.strip()


def _join_spaced_letters(line):
    """'A n h y d r i t e' -> 'Anhydrite' (runs of single characters become one word)."""
    out, run = [], []
    for tok in line.split():
        if len(tok) == 1:
            run.append(tok)
        else:
            if run: out.append("".join(run)); run = []
            out.append(tok)
    if run: out.append("".join(run))
    return " ".join(out)


def ocr_below(img, bb, limit_y, u, scale=3):
    """OCR the block of text under a frame (any number of lines) as ONE label."""
    H, W = img.shape[:2]
    x, y, w, h = [int(v) for v in bb]
    x0, x1 = max(int(x - 0.3 * w), 0), min(int(x + 1.3 * w), W)
    y0, y1 = int(y + h + 0.15 * u), int(min(y + h + 3.6 * u, limit_y - 2))
    if y1 - y0 < u or x1 <= x0:
        return ""
    g = cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    g = cv2.copyMakeBorder(g, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=255)
    g = cv2.resize(g, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    txt = pytesseract.image_to_string(g, config="--psm 6")
    lines = [_join_spaced_letters(re.sub(r"[^A-Za-z0-9 .()\-]", " ", l)).strip() for l in txt.splitlines()]
    lines = [l for l in lines if sum(c.isalpha() for c in l) >= 3]
    return " ".join(lines)


def refine_box_names(img, pairs, labels, names):
    """Name each frame from OCR of the text under it; fall back to the first-pass label."""
    if not pairs or pytesseract is None:
        return pairs
    u = text_unit(labels)
    boxes = [b for b, _ in pairs]
    H = img.shape[0]
    out = []
    for b, first in pairs:
        bx, by, bw, bh = b
        below = [o[1] for o in boxes if o[1] > by + bh * 0.5 and abs((o[0] + o[2] / 2) - (bx + bw / 2)) < 0.6 * bw]
        limit = min(below) if below else H
        txt = ocr_below(img, b, limit, u)
        out.append((b, snap_name(txt or first, names)))
    return out


def detect(img, labels, layout="auto", names=None):
    """Run the chosen strategy. Returns ([(bbox, label), ...], strategy_description)."""
    names = names or KNOWN_NAMES
    ev = left_evidence(img, labels) if labels else 0.0
    if layout == "left" or (layout == "auto" and ev >= 0.6):
        p, _ = strategy_left(img, labels)
        p = [(b, snap_name(n, names)) for b, n in p]
        return p, f"left (ink beside {ev:.0%} of labels)"
    p = refine_box_names(img, strategy_boxes(img, labels), labels, names)
    return p, f"boxes (ink beside {ev:.0%} of labels)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image"); ap.add_argument("out_dir")
    ap.add_argument("--layout", choices=["auto", "left", "boxes"], default="auto")
    ap.add_argument("--no-ocr", action="store_true")
    ap.add_argument("--names", help="text file with extra known lithology names, one per line")
    ap.add_argument("--pad", type=int, default=0)
    a = ap.parse_args()

    img = cv2.imread(a.image)
    if img is None:
        sys.exit(f"cannot read {a.image}")
    labels = ocr_labels(img) if (pytesseract and not a.no_ocr) else []

    pairs, used = detect(img, labels, a.layout, load_names(a.names))

    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.png"):
        old.unlink()
    dbg = img.copy(); rows = []; seen = {}
    H, W = img.shape[:2]
    # unnamed swatches are not patterns (empty frames, stray boxes): don't save them
    skipped = [b for b, n in pairs if not n]
    pairs = [(b, n) for b, n in pairs if n]
    for x, y, w, h in skipped:
        cv2.rectangle(dbg, (x, y), (x + w, y + h), (160, 160, 160), 1)
    for i, ((x, y, w, h), name) in enumerate(pairs):
        p = a.pad
        crop = img[max(y - p, 0):min(y + h + p, H), max(x - p, 0):min(x + w + p, W)]
        base = slug(name, f"swatch_{i:03d}")
        seen[base] = seen.get(base, 0) + 1
        fn = base if seen[base] == 1 else f"{base}_{seen[base]}"
        cv2.imwrite(str(out / f"{fn}.png"), crop)
        rows.append([fn + ".png", name, x, y, w, h])
        cv2.rectangle(dbg, (x, y), (x + w, y + h), (0, 0, 255), 1)
    for l in labels:
        cv2.rectangle(dbg, (int(l["x"]), int(l["y"])), (int(l["x"] + l["w"]), int(l["y"] + l["h"])), (0, 160, 0), 1)
    cv2.imwrite(str(out / "_debug.png"), dbg)
    with open(out / "_index.csv", "w", newline="") as f:
        csv.writer(f).writerows([["file", "label", "x", "y", "w", "h"]] + rows)
    print(f"strategy={used}: {len(rows)} saved, {len(skipped)} unnamed skipped, {len(labels)} labels -> {out}")


if __name__ == "__main__":
    main()