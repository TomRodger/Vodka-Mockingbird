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
     and each box is paired with the nearest label. Labels are also re-read
     directly from the strip under each box, so multi-line labels and labels
     the global OCR pass missed are still recovered.

Usage:
  python legend_extractor.py legend.png out_dir [--layout auto|left|boxes] [--no-ocr] [--keep-unnamed]
  python legend_extractor.py full_log.tif out_dir [--annotation legend_box.json]
      (no annotation + a full page -> legend_finder.py locates the legend first)

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

OCR_ENABLED = True   # set False by --no-ocr; also disables the per-box label re-read


# ----------------------------------------------------------------------------
# OCR
# ----------------------------------------------------------------------------
def _alpha(s):
    return sum(c.isalpha() for c in s)


# Single words that are complete rock names on their own. Used only to stop the wrapped-label
# merge from gluing tightly stacked legend ROWS together (Callaway: "ANHYDRITE" / "BENTONITE" /
# "BRECCIA" became one label with one swatch box, so the orange Anhydrite swatch was lost).
ROCK_WORDS = {
    "anhydrite", "bentonite", "breccia", "cement", "chalk", "chert", "claystone", "coal",
    "conglomerate", "dolomite", "granite", "gypsum", "halite", "igneous", "kaolinite",
    "limestone", "limonite", "marl", "marlstone", "metamorphic", "pyrite", "salt", "sandstone",
    "shale", "siderite", "siltstone", "slate", "sylvite", "till", "tuff",
}


def _rock_word(word):
    """A complete rock name, allowing an OCR misread ("ILTSTONE", "Syivite")."""
    import difflib
    w = re.sub(r"[^a-z]", "", word.lower())
    if len(w) < 4:
        return False
    return w in ROCK_WORDS or bool(difflib.get_close_matches(w, ROCK_WORDS, n=1, cutoff=0.85))


def _separate_rows(upper, lower):
    """True if `lower` is the next legend row, not the second line of `upper`'s name.
    A wrapped name breaks before its rock word ("Silty" / "Sandstone", "Dolomitic" /
    "limestone", "Sandy and calcareous" / "dolomite"), so its upper line holds no complete rock
    word yet. Stacked rows are whole names each: the upper line already contains a rock word
    ("ANHYDRITE", "SHALE GRAY") and so does the lower ("BENTONITE", "SHALE COLORED")."""
    parts = upper["parts"] if "parts" in upper else [upper]
    up_has_rock = any(_rock_word(w["text"]) for p in parts for w in p["words"])
    # a name has ONE rock noun, so a full rock word on both lines means two entries
    # (Callaway: "SILTSTONE" / "SILTY SHALE", where the lower line starts with an adjective)
    return up_has_rock and any(_rock_word(w["text"]) for w in lower["words"])


JOINERS = {"or", "and", "of", "with", "not"}   # short real words in names ("SIDERITE or LIMONITE")
# Short lithology terms and qualifiers (general geology vocabulary). Like rock words, they are
# never treated as pattern junk even when OCR is unsure of them ("Argill and calc sand").
NAME_WORDS = {"sand", "silt", "clay", "calc", "lime", "dolo", "sandy", "silty", "shaly",
              "limy", "argill", "fine", "coarse", "medium", "red", "gray", "grey", "green", "black"}


def _repetitive(text):
    """OCR of a periodic hatch pattern repeats a few characters ("ateatatatatats")."""
    t = re.sub(r"[^a-z]", "", text.lower())
    return len(t) >= 6 and len(set(t)) / len(t) < 0.4


def _pattern_junk(w):
    """Text read off a swatch's pattern: low OCR confidence AND short or repetitive. Measured on
    the Rio San Gabriel / Callaway legends: such junk scores 25-54 ("cote", "aes", "NNN", "=");
    real words almost all score >= 61, and the one lower real word (LIMESTONE, 51) is long and
    not repetitive, so it is kept."""
    t = w["text"]
    if (w["conf"] >= 60 or t.lower() in JOINERS or t == "-" or _rock_word(t)        # TILL, COAL
            or re.sub(r"[^a-z]", "", t.lower()) in NAME_WORDS):
        return False
    return _alpha(t) < 5 or _repetitive(t)


def _plausible_label(text):
    """A label made of real words, not pattern junk: a word of 4+ letters, and not repetitive."""
    return _alpha(text) >= 4 and not _repetitive(text)


def ocr_labels(img, scale=None, min_conf=25, merge_gap=0.7):
    """Text labels as dicts(text,x,y,w,h,lh) in original pixel coords.

    Wrapped (multi-line) labels are merged into ONE label: lines that sit
    directly under each other (gap < merge_gap * text height) and overlap
    horizontally. `lh` is the height of a single text line.
    """
    H, W = img.shape[:2]
    if scale is None:   # small images are upscaled 3x; big ones less (cap ~3000 px)
        scale = 3.0 if max(H, W) <= 1500 else max(1.0, 3000.0 / max(H, W))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Tesseract's sparse-text pass is sensitive to scale: a label found at 3x can vanish at 2x or 4x
    # (and vice versa). Run a few scales and take the union of the words.
    words = []
    for si, sc in enumerate((scale, 0.67 * scale, 1.33 * scale)):
        big = cv2.resize(gray, None, fx=sc, fy=sc, interpolation=cv2.INTER_CUBIC)
        d = pytesseract.image_to_data(big, config="--psm 11", output_type=pytesseract.Output.DICT)
        for i, t in enumerate(d["text"]):
            t = t.strip()
            # extra scales only ADD confident words (they also read hatch patterns as junk text)
            if not t or float(d["conf"][i]) < (min_conf if si == 0 else 80):
                continue
            x, y, w, h = (d[k][i] / sc for k in ("left", "top", "width", "height"))
            cx, cy = x + w / 2, y + h / 2
            if si and any(min(x + w, o["x"] + o["w"]) > max(x, o["x"]) and
                          min(y + h, o["y"] + o["h"]) > max(y, o["y"]) for o in words):
                continue                                   # already read at the first scale
            words.append(dict(text=t, x=x, y=y, w=w, h=h, cy=cy, conf=float(d["conf"][i])))
    if not words:
        return []

    # reference text height from words that look like real words, then discard
    # tokens whose height is far off it (junk read from hatch patterns is tall or tiny).
    # Bounds are deliberately loose: a label sitting next to a pattern gets a distorted box.
    real = [w["h"] for w in words if w["conf"] >= 50 and _alpha(w["text"]) >= 4]
    ref = float(np.median(real if len(real) >= 3 else [w["h"] for w in words]))
    words = [w for w in words if w["text"] in ("-", "–", "—") or 0.4 * ref <= w["h"] <= 2.2 * ref]
    # Drop text OCR read off a swatch's pattern before chaining words into labels: otherwise it
    # bridges the gap between one label and the next column's label (Rio San Gabriel:
    # "SIDERITE or LIMONITE cote LIMESTONE", "MARLSTONE = - - - = = CLAYSTONE").
    words = [w for w in words if not _pattern_junk(w)]
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
    chains = []
    for row in rows:
        row.sort(key=lambda w: w["x"])
        chain = [row[0]]
        for wd in row[1:]:
            prev = chain[-1]
            if wd["x"] - (prev["x"] + prev["w"]) < 0.9 * med_h:
                chain.append(wd)
            else:
                chains.append(chain); chain = [wd]
        chains.append(chain)

    # 3) chain -> line box (trim short junk tokens off the left end)
    lines = []
    for ch in chains:
        while len(ch) > 1 and _alpha(ch[0]["text"]) < 4 and ch[0]["text"] != "-":
            if len(ch[0]["text"]) <= 2 and ch[0]["x"] + ch[0]["w"] < ch[1]["x"] - 0.5 * med_h:
                ch = ch[1:]
            else:
                break
        x1 = min(w["x"] for w in ch); y1 = min(w["y"] for w in ch)
        x2 = max(w["x"] + w["w"] for w in ch); y2 = max(w["y"] + w["h"] for w in ch)
        lines.append(dict(words=ch, x=x1, y=y1, w=x2 - x1, h=y2 - y1))

    # short candidates (1-2 letters/digits, one confident word): only accepted at the end,
    # and only if they line up with the real labels (hatch junk won't)
    weak = []
    for ln in lines:
        if len(ln["words"]) == 1:
            t = ln["words"][0]["text"]
            if t.isalnum() and len(t) <= 2 and ln["words"][0]["conf"] >= 60:
                weak.append(ln)

    # drop pure-junk lines (hatch/dash patterns misread as symbols). With the lower min_conf
    # they would otherwise be glued onto the real label below them by the merge step.
    lines = [ln for ln in lines if sum(_alpha(w["text"]) for w in ln["words"]) >= 2]

    # 4) merge wrapped labels ("Silty" / "Sandstone") BEFORE judging whether a label is real
    lines.sort(key=lambda l: (l["y"], l["x"]))
    merged = []
    for ln in lines:
        target = None
        for m in merged:
            gap = ln["y"] - (m["y"] + m["h"])
            ov = min(m["x"] + m["w"], ln["x"] + ln["w"]) - max(m["x"], ln["x"])
            if _separate_rows(m, ln):
                continue
            if -0.1 * med_h <= gap <= merge_gap * med_h and ov > 0.4 * min(m["w"], ln["w"]):
                target = m
                break
        if target is None:
            merged.append(dict(ln, parts=[ln]))
        else:
            target["parts"].append(ln)
            x1 = min(target["x"], ln["x"]); y1 = target["y"]
            x2 = max(target["x"] + target["w"], ln["x"] + ln["w"])
            y2 = max(target["y"] + target["h"], ln["y"] + ln["h"])
            target.update(x=x1, y=y1, w=x2 - x1, h=y2 - y1)

    # 5) a real label contains at least one real word somewhere in it: >=4 letters, or a short
    #    3-letter word (DST, FIT, Gaz) that OCR read with high confidence
    out = []
    for m in merged:
        if not any(_alpha(w["text"]) >= 4 or (_alpha(w["text"]) >= 3 and w["conf"] >= 80)
                   for p in m["parts"] for w in p["words"]):
            continue
        text = " ".join(w["text"] for p in m["parts"] for w in p["words"])
        lh = float(np.median([p["h"] for p in m["parts"]]))
        out.append(dict(text=text, x=m["x"], y=m["y"], w=m["w"], h=m["h"], lh=lh))

    # 6) accept short labels ("S", "Ls", "5") when they share a left edge or centre line
    #    with at least 2 real labels (i.e. they sit in the same label column)
    tol = max(4.0, med_h)
    strong = list(out)
    for ln in weak:
        if not 0.5 * med_h <= ln["h"] <= 1.8 * med_h:
            continue
        if any(min(ln["x"] + ln["w"], l["x"] + l["w"]) > max(ln["x"], l["x"]) and
               min(ln["y"] + ln["h"], l["y"] + l["h"]) > max(ln["y"], l["y"]) for l in strong):
            continue                                   # already part of a real label
        lcx = ln["x"] + ln["w"] / 2
        n_left = sum(abs(ln["x"] - l["x"]) < tol for l in strong)
        n_mid = sum(abs(lcx - (l["x"] + l["w"] / 2)) < tol for l in strong)
        if max(n_left, n_mid) >= 2:
            out.append(dict(text=ln["words"][0]["text"], x=ln["x"], y=ln["y"], w=ln["w"], h=ln["h"], lh=ln["h"]))

    # 7) tesseract's page pass often returns nothing at all for 1-2 character text, so also look
    #    for small ink blobs that sit on the left edge of a label column and read each on its own
    if len(strong) >= 2:
        mask = ink_mask(img)
        for l in out:                                   # paint out text already found
            mask[max(int(l["y"]) - 2, 0):int(l["y"] + l["h"]) + 3, max(int(l["x"]) - 2, 0):int(l["x"] + l["w"]) + 3] = 0
        k = max(3, int(0.35 * med_h))
        glued = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
        n, _, st, _ = cv2.connectedComponentsWithStats(glued, connectivity=8)
        for i in range(1, n):
            bx, by, bw, bh, _a = st[i]
            bx, by, bw, bh = bx + k // 2, by + k // 2, max(bw - k, 1), max(bh - k, 1)   # undo dilation growth
            if not (0.5 * med_h <= bh <= 1.8 * med_h and bw <= 2.5 * bh):
                continue
            if sum(abs(bx - l["x"]) < tol for l in strong) < 2:
                continue
            if any(abs(bx - l["x"]) < tol and abs((by + bh / 2) - (l["y"] + l["h"] / 2)) < 0.7 * med_h for l in out):
                continue
            t, c = _read_short(img, bx, by, bw, bh)
            if t and c >= 50:
                out.append(dict(text=t, x=float(bx), y=float(by), w=float(bw), h=float(bh), lh=float(bh)))
    return out


def _read_short(img, x, y, w, h, pad=4):
    """OCR a tiny crop as ONE word (psm 8; psm 10 for a single glyph). Returns (text, conf)."""
    H, W = img.shape[:2]
    x0, y0, x1, y1 = max(int(x) - pad, 0), max(int(y) - pad, 0), min(int(x + w) + pad, W), min(int(y + h) + pad, H)
    g = cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    sc = float(np.clip(40.0 / max(h, 1.0), 1.0, 6.0))
    g = cv2.resize(g, None, fx=sc, fy=sc, interpolation=cv2.INTER_CUBIC)
    g = cv2.copyMakeBorder(g, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)
    best = ("", 0.0)
    for psm in (8, 10):
        d = pytesseract.image_to_data(g, config=f"--psm {psm}", output_type=pytesseract.Output.DICT)
        for t, c in zip(d["text"], d["conf"]):
            t = t.strip()
            if t.isalnum() and len(t) <= 2 and float(c) > best[1]:
                best = (t, float(c))
    return best


def ocr_region(img, x0, y0, x1, y1, u=12.0):
    """Read the text inside a window (psm 6 = block of text, so multi-line labels stay together)."""
    if pytesseract is None or not OCR_ENABLED:
        return ""
    H, W = img.shape[:2]
    x0, y0, x1, y1 = max(int(x0), 0), max(int(y0), 0), min(int(x1), W), min(int(y1), H)
    if x1 - x0 < 10 or y1 - y0 < 6:
        return ""
    gray = cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    s = float(np.clip(40.0 / max(u, 1.0), 1.0, 4.0))          # aim for ~40 px tall text
    big = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
    big = cv2.copyMakeBorder(big, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)
    txt = pytesseract.image_to_string(big, config="--psm 6")
    toks = []
    for line in txt.splitlines():
        for t in re.sub(r"[^A-Za-z0-9()/,.&\- ]+", " ", line).split():
            if sum(c.isalnum() for c in t) >= 1:
                toks.append(t)
    if len(toks) == 1 and len(toks[0]) <= 2 and toks[0].isalnum():
        return toks[0]                            # a short code label ("S", "Ls")
    if not any(_alpha(t) >= 4 for t in toks):     # nothing word-like: it was hatch/dash junk
        return ""
    return " ".join(toks)


def text_unit(labels, default=10.0):
    """Typical height of ONE text line in px: every pixel threshold below scales with this."""
    return float(np.median([l.get("lh", l["h"]) for l in labels])) if labels else default


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
        # ...but a short column whose every entry is a real word is a real column too: a legend
        # with a 2-row rock grid beside an 8-row symbol list (Shell AC0336) lost all 8 rocks.
        cols = [c for c in cols if len(c["items"]) >= max(3, min_frac * biggest)
                or (len(c["items"]) >= 2 and all(_plausible_label(i["text"]) for i in c["items"]))]
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
        def left_bound(cy):
            # A swatch can't reach into the label beside it, but only labels on the SAME row
            # bound it: a block of other entries above or below (ocs-g 16987's drilling symbols,
            # 200 rows lower and offset) must not shrink the search window to nothing.
            ends = [o["x"] + o["w"] for cj in range(ci) for o in cols[cj]["items"]
                    if abs((o["y"] + o["h"] / 2) - cy) < pitch / 2 and o["x"] + o["w"] <= c["x"]]
            return max(ends) + 2 if ends else 0

        for l in items:
            cy = l["y"] + l["h"] / 2
            x0 = max(left_bound(cy), l["x"] - max_swatch_w)
            bb = ink_bbox(mask, x0, cy - pitch / 2 + 1, l["x"] - 2, cy + pitch / 2 - 1)
            found.append((l, bb))

        # fill rows OCR missed (gap bigger than ~1.6 pitch) as label-less rows
        for a, b in zip(items[:-1], items[1:]):
            gap = (b["y"] + b["h"] / 2) - (a["y"] + a["h"] / 2)
            n_missing = int(round(gap / pitch)) - 1
            for k in range(1, n_missing + 1):
                cy = a["y"] + a["h"] / 2 + k * gap / (n_missing + 1)
                fake = dict(text="", x=c["x"], y=cy - a["h"] / 2, w=a["w"], h=a["h"])
                x0 = max(left_bound(cy), c["x"] - max_swatch_w)
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


def label_window(b, boxes, u):
    """Strip of the image directly under a framed swatch where its label should be."""
    bx, by, bw, bh = b
    pad = 0.15 * bw
    y0 = by + bh + 3
    y1 = y0 + 5 * u                                   # room for up to ~2-3 wrapped lines
    for o in boxes:                                   # stop at the next swatch below in this column
        if o is not b and o[1] > by + bh and o[0] < bx + bw and o[0] + o[2] > bx:
            y1 = min(y1, o[1] - 2)
    return bx - pad, y0, bx + bw + pad, y1


def strategy_boxes(img, labels, dilate=None, min_w=None):
    """Framed swatches: find rectangles by their outline, then name them from nearby labels.

    OCR is NOT used to find swatches (its boxes can clip frames and its junk tokens
    mislead); a frame is any blob with ink along >=3 of its 4 sides -- text blobs
    don't have that.

    Names: the global OCR label nearest below/right of the frame, but the strip under
    the frame is also re-read on its own (multi-line aware). The reading with more
    letters wins, so a label the global pass dropped or truncated is recovered.
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
        g_name = labels[best]["text"] if best is not None else ""

        # second opinion: re-read the strip under the frame (catches dropped / wrapped labels)
        r_name = ocr_region(img, *label_window(b, boxes, u), u=u)
        name = r_name if _alpha(r_name) > _alpha(g_name) else g_name
        out.append((b, name))
    return out


# ----------------------------------------------------------------------------
def slug(s, fallback):
    return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_").lower() or fallback


def detect(img, labels, layout="auto"):
    """Run the chosen strategy. Returns ([(bbox, label), ...], strategy_description)."""
    ev = left_evidence(img, labels) if labels else 0.0
    if layout == "left" or (layout == "auto" and ev >= 0.6):
        p, _ = strategy_left(img, labels)
        return p, f"left (ink beside {ev:.0%} of labels)"
    return strategy_boxes(img, labels), f"boxes (ink beside {ev:.0%} of labels)"


# ----------------------------------------------------------------------------
# Locating the legend inside a full log page
# ----------------------------------------------------------------------------
def _read_image(path):
    """cv2 first; PIL fallback for big / unusual TIFFs (first page)."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is not None:
        return img
    try:
        from PIL import Image
        Image.MAX_IMAGE_PIXELS = None
        im = Image.open(str(path)); im.seek(0)
        return cv2.cvtColor(np.array(im.convert("RGB")), cv2.COLOR_RGB2BGR)
    except Exception:
        return None


def load_annotation_bbox(path):
    """Legend box (x, y, w, h) in pixels from an annotation JSON file.

    Accepted: [x, y, w, h]; {"x","y","w","h"}; {"x1","y1","x2","y2"}; {"xmin","ymin","xmax","ymax"};
    {"left","top","right","bottom"}; any of these nested under "bbox"/"box"/"legend"/"region";
    or a list of such objects (the one mentioning "legend" wins, else the first).
    """
    import json
    d = json.load(open(path, encoding="utf-8"))

    def one(o):
        if isinstance(o, (list, tuple)) and len(o) == 4 and all(isinstance(v, (int, float)) for v in o):
            return tuple(float(v) for v in o)
        if isinstance(o, dict):
            for k in ("bbox", "box", "legend", "region"):
                if k in o:
                    r = one(o[k])
                    if r:
                        return r
            for ks, kind in ((("x", "y", "w", "h"), "xywh"), (("x", "y", "width", "height"), "xywh"),
                             (("x1", "y1", "x2", "y2"), "xyxy"), (("xmin", "ymin", "xmax", "ymax"), "xyxy"),
                             (("left", "top", "right", "bottom"), "xyxy")):
                if all(k in o for k in ks):
                    v = [float(o[k]) for k in ks]
                    return tuple(v) if kind == "xywh" else (v[0], v[1], v[2] - v[0], v[3] - v[1])
        return None

    if isinstance(d, list) and d and isinstance(d[0], dict):
        d = sorted(d, key=lambda o: 0 if "legend" in json.dumps(o).lower() else 1)[0]
    box = one(d)
    if not box or box[2] <= 0 or box[3] <= 0:
        sys.exit(f"cannot read a legend box from {path}; expected [x,y,w,h] or {{x,y,w,h}} / {{x1,y1,x2,y2}}")
    return tuple(int(round(v)) for v in box)


def looks_like_full_page(img):
    """A whole log page is big and tall; a legend crop is small. (--find / --no-find override this.)"""
    H, W = img.shape[:2]
    return H * W >= 8_000_000 or H >= 1.6 * W


def locate_legend(img, a, out):
    """Return (legend_image, description).

    annotation given  -> crop that box.
    no annotation     -> legend_finder (CV only) if the image looks like a full page.
    The crop is written to out/_legend_crop.png and read back, so what is cut is exactly the saved image.
    """
    H, W = img.shape[:2]
    if a.annotation:
        x, y, w, h = load_annotation_bbox(a.annotation)
        x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, W), min(y + h, H)
        if x1 <= x0 or y1 <= y0:
            sys.exit(f"annotation box {x, y, w, h} is outside the {W}x{H} image")
        crop, how = img[y0:y1, x0:x1], f"legend from annotation: x={x0} y={y0} w={x1 - x0} h={y1 - y0}"
    else:
        if not (a.find or (not a.no_find and looks_like_full_page(img))):
            return img, "no annotation; image looks like a cropped legend already (use --find to search it)"
        try:
            import legend_finder
        except ImportError:
            sys.exit("legend_finder.py must be in the same folder as legend_extractor.py")
        res = legend_finder.find_legend(img)
        if res is None:
            sys.exit("legend_finder could not find a legend near the top of the page; "
                     "pass --annotation <file> (or crop the legend yourself)")
        x, y, w, h = res["bbox"]
        crop = res["crop"]
        how = (f"legend found by legend_finder: x={x} y={y} w={w} h={h} "
               f"({res['n_swatches']} swatches, confidence {res['confidence']:.2f})")
    path = out / "_legend_crop.png"
    cv2.imwrite(str(path), crop)
    back = cv2.imread(str(path), cv2.IMREAD_COLOR)
    return (back if back is not None else crop), how


def main():
    global OCR_ENABLED
    ap = argparse.ArgumentParser()
    ap.add_argument("image"); ap.add_argument("out_dir")
    ap.add_argument("--layout", choices=["auto", "left", "boxes"], default="auto")
    ap.add_argument("--no-ocr", action="store_true")
    ap.add_argument("--keep-unnamed", action="store_true",
                    help="save swatches without a label as swatch_NNN instead of skipping them (debugging)")
    ap.add_argument("--pad", type=int, default=0)
    ap.add_argument("--annotation", help="JSON file with the legend box; if omitted, legend_finder looks for it")
    ap.add_argument("--find", action="store_true", help="always run legend_finder (even on a small image)")
    ap.add_argument("--no-find", action="store_true", help="never run legend_finder; use the image as the legend")
    a = ap.parse_args()
    OCR_ENABLED = not a.no_ocr

    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.png"):
        old.unlink()

    img = _read_image(a.image)
    if img is None:
        sys.exit(f"cannot read {a.image}")
    img, how = locate_legend(img, a, out)
    print(how)
    labels = ocr_labels(img) if (pytesseract and not a.no_ocr) else []

    pairs, used = detect(img, labels, a.layout)

    dbg = img.copy(); rows = []; seen = {}
    H, W = img.shape[:2]
    # unnamed swatches are not patterns (empty frames, stray boxes): don't save them
    skipped = [] if a.keep_unnamed else [b for b, n in pairs if not n]
    if not a.keep_unnamed:
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