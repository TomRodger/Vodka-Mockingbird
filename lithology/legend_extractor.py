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
    import pytesseract
except ImportError:
    pytesseract = None


# ----------------------------------------------------------------------------
# OCR
# ----------------------------------------------------------------------------
def ocr_labels(img, scale=3, min_conf=40):
    """Text *lines* as dicts(text,x,y,w,h) in original pixel coords."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    big = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    d = pytesseract.image_to_data(big, config="--psm 11", output_type=pytesseract.Output.DICT)
    words = []
    for i, t in enumerate(d["text"]):
        t = t.strip()
        if not t or float(d["conf"][i]) < min_conf:
            continue
        x, y, w, h = (d[k][i] / scale for k in ("left", "top", "width", "height"))
        words.append(dict(text=t, x=x, y=y, w=w, h=h, cy=y + h / 2))
    if not words:
        return []

    # 1) group words into text lines by vertical centre
    med_h = float(np.median([w["h"] for w in words]))
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

    # 3) a real label contains at least one real word (>=4 letters); this drops
    #    the junk tokens OCR invents from the hatch patterns
    out = []
    for ch in lines:
        if not any(sum(c.isalpha() for c in w["text"]) >= 4 for w in ch):
            continue
        # trim leading junk tokens (tiny, non-word) glued in front of the label
        while len(ch) > 1 and sum(c.isalpha() for c in ch[0]["text"]) < 4 and ch[0]["text"] not in ("-",):
            if len(ch[0]["text"]) <= 2 and ch[0]["x"] + ch[0]["w"] < ch[1]["x"] - 0.5 * med_h:
                ch = ch[1:]
            else:
                break
        x1 = min(w["x"] for w in ch); y1 = min(w["y"] for w in ch)
        x2 = max(w["x"] + w["w"] for w in ch); y2 = max(w["y"] + w["h"] for w in ch)
        out.append(dict(text=" ".join(w["text"] for w in ch), x=x1, y=y1, w=x2 - x1, h=y2 - y1))
    return out


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
def strategy_left(img, labels, max_swatch_w=70):
    """Returns (list of (bbox, label_text), ink_fraction)."""
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
def strategy_boxes(img, labels, dilate=5, min_size=10):
    mask = ink_mask(img)
    for l in labels:  # paint text out
        mask[max(int(l["y"] - 2), 0):int(l["y"] + l["h"] + 2),
             max(int(l["x"] - 2), 0):int(l["x"] + l["w"] + 2)] = 0
    glued = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (dilate, dilate)))
    n, _, st, _ = cv2.connectedComponentsWithStats(glued, connectivity=8)
    H, W = mask.shape
    boxes = []
    for i in range(1, n):
        x, y, w, h, _a = st[i]
        if w < min_size or h < min_size or w * h > 0.5 * W * H:
            continue
        bb = ink_bbox(mask, x, y, x + w, y + h)
        if bb:
            boxes.append(bb)
    if len(boxes) >= 4:  # drop size outliers (title bars, stray marks)
        mw = np.median([b[2] for b in boxes]); mh = np.median([b[3] for b in boxes])
        boxes = [b for b in boxes if 0.5 * mw <= b[2] <= 1.8 * mw and 0.5 * mh <= b[3] <= 1.8 * mh]

    used, out = set(), []
    for b in sorted(boxes, key=lambda b: (b[1] // 10, b[0])):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image"); ap.add_argument("out_dir")
    ap.add_argument("--layout", choices=["auto", "left", "boxes"], default="auto")
    ap.add_argument("--no-ocr", action="store_true")
    ap.add_argument("--pad", type=int, default=0)
    a = ap.parse_args()

    img = cv2.imread(a.image)
    if img is None:
        sys.exit(f"cannot read {a.image}")
    labels = ocr_labels(img) if (pytesseract and not a.no_ocr) else []

    pairs, used = [], None
    if a.layout in ("auto", "left") and labels:
        p, frac = strategy_left(img, labels)
        if a.layout == "left" or frac >= 0.6:
            pairs, used = p, f"left (ink in {frac:.0%} of rows)"
    if used is None:
        pairs, used = strategy_boxes(img, labels), "boxes"

    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.png"):
        old.unlink()
    dbg = img.copy(); rows = []; seen = {}
    H, W = img.shape[:2]
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
    print(f"strategy={used}: {len(rows)} swatches, {len(labels)} labels -> {out}")


if __name__ == "__main__":
    main()