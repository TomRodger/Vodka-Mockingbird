#!/usr/bin/env python3
"""
Sanity test for legend_cutter on an unseen legend (no ground truth needed).

Usage:
  python test_legend.py legend.png [--expected N] [--layout auto|left|boxes] [--out DIR]

FAIL  = output is wrong/unusable      -> exit code 1
WARN  = suspicious, look at the overlay
Always writes DIR/_test_overlay.png (red = swatches, green = labels,
blue = ink the cutter did NOT account for, i.e. likely missed swatches).
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

import legend_extractor as lc


def iou(a, b):
    ax, ay, aw, ah = a; bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    return inter / (aw * ah + bw * bh - inter + 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--expected", type=int, help="exact swatch count if you know it")
    ap.add_argument("--layout", default="auto", choices=["auto", "left", "boxes"])
    ap.add_argument("--out", default="test_out")
    a = ap.parse_args()

    fails, warns = [], []
    img = cv2.imread(a.image)
    if img is None:
        sys.exit(f"FAIL: cannot read {a.image}")
    H, W = img.shape[:2]

    labels = lc.ocr_labels(img) if lc.pytesseract else []
    if not lc.pytesseract:
        warns.append("pytesseract not installed: running without labels")
    pairs, strategy = lc.detect(img, labels, a.layout)
    boxes = [b for b, _ in pairs]
    names = [n for _, n in pairs]
    print(f"image {W}x{H} | strategy: {strategy} | {len(boxes)} swatches | {len(labels)} OCR labels")

    # ---- hard checks -------------------------------------------------------
    if not boxes:
        fails.append("no swatches found")
    else:
        # 1. inside the image
        oob = [b for b in boxes if b[0] < 0 or b[1] < 0 or b[0] + b[2] > W or b[1] + b[3] > H]
        if oob:
            fails.append(f"{len(oob)} swatch box(es) extend outside the image")

        # 2. no two swatches should overlap
        overl = [(i, j) for i in range(len(boxes)) for j in range(i + 1, len(boxes))
                 if iou(boxes[i], boxes[j]) > 0.2]
        if overl:
            fails.append(f"{len(overl)} overlapping swatch pair(s), e.g. #{overl[0][0]} & #{overl[0][1]} (merged or double-detected)")

        # 3. swatches in one legend are about the same size
        def on_edge(b):
            return b[0] <= 1 or b[1] <= 1 or b[0] + b[2] >= W - 2 or b[1] + b[3] >= H - 2
        edge = [i for i, b in enumerate(boxes) if on_edge(b)]
        if edge:
            warns.append(f"{len(edge)} swatch(es) touch the image edge (probably cut off in the source): #{edge[:8]}")
        inner = [b for i, b in enumerate(boxes) if i not in edge] or boxes
        ws = np.array([b[2] for b in inner]); hs = np.array([b[3] for b in inner])
        mw, mh = np.median(ws), np.median(hs)
        odd = [i for i, b in enumerate(boxes) if i not in edge
               and not (0.6 * mw <= b[2] <= 1.6 * mw and 0.6 * mh <= b[3] <= 1.6 * mh)]
        if odd:  # warning only: legends can contain clipped or genuinely different-sized swatches
            warns.append(f"{len(odd)}/{len(boxes)} swatch(es) differ a lot in size from the median "
                         f"({mw:.0f}x{mh:.0f}), maybe clipped or merged: #{odd[:8]}")

        if a.expected is not None and len(boxes) != a.expected:
            fails.append(f"expected {a.expected} swatches, got {len(boxes)}")

        # ---- soft checks ---------------------------------------------------
        mask = lc.ink_mask(img)
        blank = []
        for i, (x, y, w, h) in enumerate(boxes):
            crop = mask[max(y, 0):y + h, max(x, 0):x + w]
            if crop.size and (crop > 0).mean() < 0.01:
                blank.append(i)
        if blank:
            warns.append(f"{len(blank)} blank swatch(es) (fine if the legend has empty ones): #{blank[:8]}")

        named = sum(1 for n in names if n)
        if labels and named < 0.7 * len(names):
            warns.append(f"only {named}/{len(names)} swatches got a label")
        dup = {n for n in names if n and names.count(n) > 1}
        if dup:
            warns.append(f"duplicate labels: {sorted(dup)[:5]}")
        if labels and abs(len(labels) - len(boxes)) > 0.2 * max(len(labels), len(boxes)):
            warns.append(f"{len(boxes)} swatches vs {len(labels)} labels differ by >20%")

        # 4. unaccounted ink = probably missed swatches
        resid = mask.copy()
        for x, y, w, h in boxes:
            resid[max(y - 3, 0):y + h + 3, max(x - 3, 0):x + w + 3] = 0
        for l in labels:
            resid[max(int(l["y"] - 3), 0):int(l["y"] + l["h"] + 3), max(int(l["x"] - 3), 0):int(l["x"] + l["w"] + 3)] = 0
        total = (mask > 0).sum()
        frac = (resid > 0).sum() / max(total, 1)
        n, _, st, _ = cv2.connectedComponentsWithStats(cv2.dilate(resid, np.ones((5, 5), np.uint8)))
        # swatch-shaped only: wide regions are text OCR missed, not swatches
        big = [s for s in st[1:] if 0.5 * mw <= s[2] <= 1.8 * mw and 0.3 * mh <= s[3] <= 1.8 * mh]
        if big:
            warns.append(f"{len(big)} swatch-sized ink region(s) not captured ({frac:.0%} of ink left over) -- see blue boxes")
        elif frac > 0.15:
            warns.append(f"{frac:.0%} of the ink is unaccounted for (titles/borders are normal)")

        # ---- overlay -------------------------------------------------------
        dbg = img.copy()
        for l in labels:
            cv2.rectangle(dbg, (int(l["x"]), int(l["y"])), (int(l["x"] + l["w"]), int(l["y"] + l["h"])), (0, 160, 0), 1)
        for x, y, w, h in boxes:
            cv2.rectangle(dbg, (x, y), (x + w, y + h), (0, 0, 255), 1)
        for s in big:
            cv2.rectangle(dbg, (s[0], s[1]), (s[0] + s[2], s[1] + s[3]), (255, 0, 0), 2)
        out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out / "_test_overlay.png"), dbg)
        print(f"overlay: {out / '_test_overlay.png'}")

    for f in fails: print("FAIL:", f)
    for w in warns: print("WARN:", w)
    if not fails:
        print("PASS" + (" (with warnings)" if warns else ""))
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()