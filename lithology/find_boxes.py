"""Bonus: find the three regions WITHOUT the supplied boxes, and measure how well that works.

    python -m lithology.find_boxes IMAGE [--out boxes.json]   # boxes for one scan
    python -m lithology.find_boxes --score [--only TEXT]      # every dataset page vs supplied boxes
    python -m lithology.pipeline IMAGE --no-boxes             # whole pipeline on found boxes
    python -m lithology.batch --no-boxes                      # every scan, no supplied boxes

The finders are the team's (unchanged here):
  lithology column  lithology_finder.find_lithology_column   (framed track filled with pattern)
  legend            legend_finder.find_legend                (grid of same-sized swatch frames)
  depth ruler       depth.depth_finder.find_depth_column     (vertically aligned increasing numbers)
This module only joins them: same box format as the supplied annotations ([x0, y0, x1, y1],
"bounding_boxes"), so the rest of the pipeline runs unchanged, and a finder that fails costs
that region with a reason instead of crashing the run.

--score compares each found box with the supplied one and writes out/bonus/scores.csv:
  IoU     overlap / union (1 = identical). Strict: a box that is right but has extra white
          margin scores low (example: the depth ruler found at the right height, 3x as wide,
          all 16 labels read, IoU 0.36).
  covers  share of the supplied box inside the found one: did it find the region at all.
A region counts as found if covers >= 0.9 and IoU >= 0.3 (not just the whole page).
"""

import argparse
import csv
import glob
import json
import os
import time

from .data import DATASET_ANNOTATIONS, DATASET_IMAGES, OUT_DIR, load_annotation

REGIONS = ("interpreted_lithology_column", "lithology_legend", "depth_column")


def xywh_to_xyxy(b):
    x, y, w, h = b
    return [int(x), int(y), int(x + w), int(y + h)]


def covers(found, truth):
    """Share of the supplied box that lies inside the found box."""
    ix = max(0, min(found[2], truth[2]) - max(found[0], truth[0]))
    iy = max(0, min(found[3], truth[3]) - max(found[1], truth[1]))
    area = (truth[2] - truth[0]) * (truth[3] - truth[1])
    return ix * iy / area if area > 0 else 0.0


def is_found(i, c):
    return c >= 0.9 and i >= 0.3


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _lithology(image):
    from .lithology_finder import find_lithology_column
    r = find_lithology_column(image)
    return ([int(v) for v in r["bbox"]], f"confidence {r.get('confidence', 0):.2f}") if r else (None, "no column found")


def _legend(image):
    from .legend_finder import find_legend, load_image
    r = find_legend(load_image(image))
    return (xywh_to_xyxy(r["bbox"]), f"{r.get('n_swatches', '?')} swatch frames") if r else (None, "no legend found")


def _depth(image):
    from .depth.depth_finder import find_depth_column
    r = find_depth_column(image)
    return [int(v) for v in r["bbox"]], f"{r.get('n_markers', '?')} ruler labels"


FINDERS = {"interpreted_lithology_column": _lithology, "lithology_legend": _legend, "depth_column": _depth}


def find_boxes(image):
    """{region: [x0, y0, x1, y1] or None}, {region: note}, {region: seconds}."""
    boxes, notes, secs = {}, {}, {}
    for name, finder in FINDERS.items():
        t0 = time.perf_counter()
        try:
            boxes[name], notes[name] = finder(image)
        except Exception as e:                      # one failed finder costs that region only
            boxes[name], notes[name] = None, f"{type(e).__name__}: {e}"
        secs[name] = round(time.perf_counter() - t0, 1)
    return boxes, notes, secs


def write_annotation(path, image, boxes):
    """Write found boxes in the supplied annotations' format, so the pipeline reads them as is."""
    try:
        from PIL import Image
        Image.MAX_IMAGE_PIXELS = None
        with Image.open(image) as im:
            width, height = im.size                 # header only, no pixel decode
    except Exception:
        width = height = None
    with open(path, "w") as f:
        json.dump(dict(image_name=os.path.basename(image), width=width, height=height, bbox_format="xyxy",
                       source="lithology.find_boxes",
                       bounding_boxes={k: v for k, v in boxes.items() if v is not None}), f, indent=2)


def annotation_for(image, out_path):
    """Find the boxes and write them to out_path. Raises if a region the pipeline needs is missing."""
    boxes, notes, _ = find_boxes(image)
    write_annotation(out_path, image, boxes)
    missing = [f"{k}: {notes[k]}" for k in REGIONS if boxes[k] is None]
    if missing:
        raise RuntimeError("bonus finder failed - " + "; ".join(missing))
    return out_path


def _find_in_subprocess(image, timeout=1800):
    """find_boxes in its own process: each finder loads the whole page (1.5 GB on the tallest
    log), so one long process ran out of memory on page 20. A fresh process per page frees it,
    and a page that still runs out of memory costs only itself."""
    import subprocess
    import sys
    try:
        p = subprocess.run([sys.executable, "-m", "lithology.find_boxes", image, "--json"],
                           capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        why = f"timeout after {timeout} s"
    else:
        if p.returncode == 0:
            d = json.loads(p.stdout.strip().splitlines()[-1])
            return d["boxes"], d["notes"], d["secs"]
        why = ("killed: out of memory" if p.returncode in (-9, 137)
               else (p.stderr.strip().splitlines() or ["crashed"])[-1])
    return ({k: None for k in REGIONS}, {k: why for k in REGIONS}, {k: None for k in REGIONS})


def _median_s(rows, k):
    v = sorted(float(r[k + "_s"]) for r in rows if r.get(k + "_s") not in (None, "", "None"))
    return v[len(v) // 2] if v else "-"


def score(only=None):
    """IoU of every finder against the supplied boxes on every dataset page. Resumes: pages
    already in out/bonus/scores.csv are kept, not re-run."""
    out_dir = os.path.join(OUT_DIR, "bonus")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "scores.csv")
    rows = []
    if os.path.exists(path):
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                for k in r:
                    if k.endswith(("_iou", "_covers")):
                        r[k] = float(r[k])
                rows.append(r)
    done = {r["document"] for r in rows}
    anns = sorted(glob.glob(os.path.join(DATASET_ANNOTATIONS, "*.json")))
    if only:
        anns = [a for a in anns if only in os.path.basename(a)]
    for i, ann in enumerate(anns, 1):
        stem = os.path.basename(ann)[:-5]
        imgs = [p for p in glob.glob(os.path.join(DATASET_IMAGES, glob.escape(stem) + ".*"))]
        if not imgs or stem in done:
            continue
        truth = load_annotation(ann)
        boxes, notes, secs = _find_in_subprocess(imgs[0])
        row = dict(document=stem)
        for k in REGIONS:
            row[k + "_iou"] = round(iou(boxes[k], truth[k]), 3) if boxes[k] and k in truth else 0.0
            row[k + "_covers"] = round(covers(boxes[k], truth[k]), 3) if boxes[k] and k in truth else 0.0
            row[k + "_s"] = secs[k]
            row[k + "_note"] = notes[k]
        rows.append(row)
        print(f"[{i}/{len(anns)}] {stem[:40]:40s} " + "  ".join(
            f"{('column', 'legend', 'depth')[j]} IoU {row[k + '_iou']:.2f} covers {row[k + '_covers']:.2f}"
            for j, k in enumerate(REGIONS)), flush=True)
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    print(f"\n{len(rows)} pages; found = covers >= 0.9 of the supplied box and IoU >= 0.3:")
    for k in REGIONS:
        ok = [r for r in rows if is_found(r[k + "_iou"], r[k + "_covers"])]
        mean = sum(r[k + "_iou"] for r in ok) / len(ok) if ok else 0
        print(f"  {k:30s} found {len(ok)}/{len(rows)}   mean IoU when found {mean:.3f}   "
              f"median {_median_s(rows, k)} s/page")
    every = sum(all(is_found(r[k + "_iou"], r[k + "_covers"]) for k in REGIONS) for r in rows)
    print(f"  all three found on {every}/{len(rows)} pages")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image", nargs="?", help="one scan (omit with --score)")
    ap.add_argument("--out", help="write the boxes here as an annotation JSON")
    ap.add_argument("--score", action="store_true", help="compare with the supplied boxes on every page")
    ap.add_argument("--only", help="with --score: only pages whose name contains this")
    ap.add_argument("--json", action="store_true", help="print the result as one JSON line (used by --score)")
    a = ap.parse_args()
    if a.score:
        return score(a.only)
    if not a.image:
        ap.error("give an IMAGE, or --score")
    boxes, notes, secs = find_boxes(a.image)
    if a.json:
        return print(json.dumps(dict(boxes=boxes, notes=notes, secs=secs)))
    for k in REGIONS:
        print(f"{k:30s} {boxes[k]}  ({notes[k]}, {secs[k]} s)")
    if a.out:
        write_annotation(a.out, a.image, boxes)
        print("wrote", a.out)


if __name__ == "__main__":
    main()
