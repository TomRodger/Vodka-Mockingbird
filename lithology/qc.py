"""Quality checks: say which outputs to trust, so a fully automated run can be relied on.

Nobody can hand-check thousands of scanned logs, so every output says where it may be wrong.
Only checks that were shown to point at real failures are used (see "What was tested").

Per interval (<scan>_qc.csv lists the flagged ones):
  unknown         no legend swatch close in colour, so the pipeline refused to name it
  depth_missing   a boundary's depth could not be read (past the readable ruler)
  depth_guessed   a boundary's depth is extrapolated beyond the last readable ruler label

Per document (green / amber / red, in the pipeline summary and batch_log.csv):
  legend coverage the legend is read a second, independent way (plain OCR of the whole box)
                  and every rock word found there is compared with the swatches extracted.
                  A rock shown in the legend but not extracted can be SILENTLY named as its
                  nearest look-alike, so its name is listed. This is the mechanism behind
                  every confidently-wrong name found in the labelled sections: Callaway's
                  anhydrite (swatch merged away -> named SHALY SILTSTONE), mc761's Shale
                  (missing -> 558 rows unknown), dc802's Silty clay-shale (missing -> Clay
                  Shale won by luck).
  red             fewer than 5 swatches read, >20% of rows unnamed, or >5% without depth
  amber           any rock missing from the extracted legend, a swatch name holding two rock
                  names, >2% unnamed, any row without depth, or one rock for the whole column
                  (every such case found, dev log08/log09 and test Nichols 1, had missed boundaries)
  The red/amber cut-offs are fixed in advance, not tuned.

What was tested and dropped (python -m tools.qc_eval reproduces it): per-interval flags from
the match evidence (poor shape fit, poor colour fit, near tie with the runner-up, decided by
the commoner-rock prior). On the 7 labelled sections the signals of wrong and right names
overlap completely (wrong names' shape distance at most 0.72; correct names reach 1.78), and
every row those flags marked was correctly named. They would have been false alarms.
"""

import difflib
import math
import re

import cv2
import numpy as np
from PIL import Image

from .data import load_annotation
from . import legend_extractor as lx

Image.MAX_IMAGE_PIXELS = None


def name_flags(row):
    """Flags about the rock NAME of one interval."""
    return ["unknown"] if row.get("lithology") == "unknown" else []


def depth_flags(row):
    """Flags about the DEPTHS of one interval (none for blank paper: no rock to date)."""
    if row.get("lithology") == "no_data":
        return []
    vals = [row.get("top_depth"), row.get("base_depth")]
    if any(v is None or (isinstance(v, float) and math.isnan(v)) for v in vals):
        return ["depth_missing"]
    return ["depth_guessed"] if row.get("depth_extrapolated") else []


def flags(row):
    return name_flags(row) + depth_flags(row)


def _rock(word):
    """The rock word a token stands for, allowing OCR slips and plurals ("Metamorphics")."""
    w = re.sub(r"[^a-z]", "", word.lower())
    if len(w) < 4:
        return None
    if w in lx.ROCK_WORDS:
        return w
    hit = difflib.get_close_matches(w, lx.ROCK_WORDS, n=1, cutoff=0.85)
    return hit[0] if hit else None


def _rocks_in(text):
    return {r for r in (_rock(t) for t in re.findall(r"[A-Za-z]+", text)) if r}


def legend_coverage(image, annotation, swatch_names):
    """(missing, merged): rock words the legend shows but no extracted swatch carries, and
    extracted names that hold two rock words (two legend rows glued into one)."""
    if lx.pytesseract is None:
        return [], []
    box = load_annotation(annotation)["lithology_legend"]
    g = np.asarray(Image.open(image).crop(tuple(box)).convert("L"))
    sc = 3.0 if max(g.shape) <= 1500 else 3000 / max(g.shape)
    text = lx.pytesseract.image_to_string(cv2.resize(g, None, fx=sc, fy=sc, interpolation=cv2.INTER_CUBIC),
                                          config="--psm 11")
    have = set().union(*[_rocks_in(n) for n in swatch_names]) if swatch_names else set()
    missing = sorted(_rocks_in(text) - have)
    merged = [n for n in swatch_names if len(_rocks_in(n)) >= 2 and not re.search(r"\b(or|and)\b", n.lower())]
    return missing, merged


def document_light(rows, n_swatches, missing=(), merged=()):
    """Green / amber / red for a whole document, with the reason."""
    rock = [r for r in rows if r.get("lithology") != "no_data"]
    n_rows = sum(r["bottom_y"] - r["top_y"] for r in rock) or 1
    share = lambda pred: sum(r["bottom_y"] - r["top_y"] for r in rock if pred(r)) / n_rows
    unknown = share(lambda r: r.get("lithology") == "unknown")
    no_depth = share(lambda r: "depth_missing" in depth_flags(r))
    if n_swatches < 5:
        return "red", f"only {n_swatches} legend swatches read"
    if unknown > 0.20:
        return "red", f"{unknown:.0%} of rows unnamed"
    if no_depth > 0.05:
        return "red", f"{no_depth:.0%} of rows without depth"
    why = []
    # One rock for the whole column: on every document where this happened (dev log08, log09;
    # test Nichols 1) the column visibly alternates between beds (black-and-white dashes vs
    # dots, no colour difference) and segmentation missed every boundary. No document that
    # was correct came out as a single rock, so this is a reliable "check me".
    rock_ids = {r.get("cluster_id") for r in rock if r.get("cluster_id", -1) >= 0}
    if len(rock_ids) == 1 and len(rock) >= 1:
        why.append("only one rock found in the whole column: boundaries probably missed")
    if missing:
        why.append("legend rocks not extracted (may be named as a look-alike): " + ", ".join(missing))
    if merged:
        why.append("legend rows glued together: " + "; ".join(merged))
    if unknown > 0.02:
        why.append(f"{unknown:.0%} of rows unnamed")
    if no_depth > 0:
        why.append(f"{no_depth:.1%} of rows without depth")
    return ("amber", "; ".join(why)) if why else ("green", "legend fully read, all rows named and dated")
