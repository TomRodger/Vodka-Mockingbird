"""
Label sections of an interpreted lithology column quickly.

Run from the repo root:

  python -m tools.label_tool list
      every annotated document, with its column size

  python -m tools.label_tool show STEM --y0 10335 --height 2000
      renders the section (zoomed, with a pixel ruler) and the legend into
      out/views/, and writes a DRAFT label file
      tools/labels/STEM__y10335.txt with boundaries from lithology.segment

  python -m tools.label_tool check tools/labels/STEM__y10335.txt
      validates your edited label file, writes the matching .csv, and renders
      your labels over the section so you can verify them

  python -m tools.label_tool score tools/labels/STEM__y10335.txt
      runs the whole pipeline on the labelled rows and scores it against your labels:
      boundary recall / precision (within 20 px), row ARI (clusters) and name accuracy

The draft from `show` is the PIPELINE'S OWN GUESS (boundaries and legend names), to save
typing. That biases labels towards the pipeline, so check every line against the column and
every name against the legend; a label that just accepts the draft scores 100% and means
nothing. Use --blank for boundaries only, with every name left as '?'.

STEM can be the full document name (without extension) or any unique prefix.
"""

import argparse
import colorsys
import glob
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

Image.MAX_IMAGE_PIXELS = None

from lithology.data import DATASET_ANNOTATIONS as ANN_DIR, DATASET_IMAGES as IMAGES_DIR, OUT_DIR
from lithology.segment import detect_boundaries, section_edge_margins

HERE = os.path.dirname(os.path.abspath(__file__))
LABELS_DIR = os.path.join(HERE, "labels")          # label files: committed, they're ground truth
VIEWS_DIR = os.path.join(OUT_DIR, "views")          # rendered pages: generated, git-ignored

TARGET_WIDTH = 200     # zoom narrow columns until they are about this wide on screen
PAGE_HEIGHT = 2000     # max on-screen height of one rendered page
RULER_W, GAP, NOTE_W = 80, 12, 170


# ---------------------------------------------------------------- lookup

def font(size):
    for name in ("DejaVuSans.ttf", "Arial.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def all_stems():
    return sorted(os.path.basename(p)[:-5] for p in glob.glob(f"{ANN_DIR}/*.json"))


def resolve(stem):
    stems = all_stems()
    if stem not in stems:
        matches = [s for s in stems if s.startswith(stem)]
        if len(matches) != 1:
            hint = "\n  ".join(matches) if matches else "(none) - run `list` to see names"
            sys.exit(f"'{stem}' matches {len(matches)} documents:\n  {hint}")
        stem = matches[0]
    images = [p for p in glob.glob(f"{IMAGES_DIR}/*")
              if os.path.splitext(os.path.basename(p))[0] == stem]
    if not images:
        sys.exit(f"No image found for {stem}")
    boxes = json.load(open(f"{ANN_DIR}/{stem}.json"))["bounding_boxes"]
    return stem, images[0], boxes


def label_path(stem, y0, ext):
    return os.path.join(LABELS_DIR, f"{stem}__y{y0}.{ext}")


def parse_label_name(path):
    base = os.path.splitext(os.path.basename(path))[0]
    if "__y" not in base:
        sys.exit(f"Label file name must look like STEM__y<number>.txt, got {base}")
    stem, y0 = base.rsplit("__y", 1)
    return stem, int(y0)


# ---------------------------------------------------------------- rendering

def strip(im, box, y0, y1, scale):
    c = im.crop((box[0], y0, box[2], y1)).convert("RGB")
    return c.resize((c.width * scale, c.height * scale), Image.Resampling.NEAREST)


def render_pages(im, boxes, y0, y1, out_base, lines=(), bands=()):
    """Draw [ruler | depth column | lith column | notes] pages.

    lines: [(y, text, colour)] horizontal marks in full-image pixel rows.
    bands: [(top, bottom, name, colour)] coloured bars beside the lith column.
    """
    lith, depth = boxes["interpreted_lithology_column"], boxes["depth_column"]
    scale = max(1, round(TARGET_WIDTH / (lith[2] - lith[0])))
    rows_per_page = max(50, PAGE_HEIGHT // scale)
    f_small, f_big = font(13), font(15)
    paths = []
    for p, ps in enumerate(range(y0, y1, rows_per_page)):
        pe = min(ps + rows_per_page, y1)
        d = strip(im, depth, ps, pe, scale)
        l = strip(im, lith, ps, pe, scale)
        W = RULER_W + d.width + GAP + l.width + GAP + 24 + NOTE_W
        page = Image.new("RGB", (W, l.height), "white")
        xd = RULER_W
        xl = xd + d.width + GAP
        xb = xl + l.width + GAP
        page.paste(d, (xd, 0))
        page.paste(l, (xl, 0))
        draw = ImageDraw.Draw(page)
        for y in range((ps // 10) * 10, pe, 10):          # ruler: tick every 10 rows
            if y < ps:
                continue
            yy = (y - ps) * scale
            big = y % 50 == 0
            draw.line([(RULER_W - (18 if big else 8), yy), (RULER_W - 2, yy)], fill=(90, 90, 90))
            if big:
                draw.text((4, yy), str(y), fill=(0, 0, 0), font=f_small, anchor="lm")
        for top, bot, name, col in bands:
            if bot <= ps or top >= pe:
                continue
            a, b = (max(top, ps) - ps) * scale, (min(bot, pe) - ps) * scale
            draw.rectangle([xb, a, xb + 20, max(a, b - 1)], fill=col)
            draw.text((xb + 26, (a + b) / 2), name, fill=(0, 0, 0), font=f_big, anchor="lm")
        for y, text, col in lines:
            if ps <= y < pe:
                yy = (y - ps) * scale
                draw.line([(xl - 4, yy), (xl + l.width + 4, yy)], fill=col, width=2)
                if text:
                    draw.text((xb + 26, yy), text, fill=col, font=f_big, anchor="lm",
                              stroke_width=2, stroke_fill=(255, 255, 255))
        path = f"{out_base}_p{p + 1:02d}.png"
        page.save(path)
        paths.append(path)
    return paths, scale


def section_dir(stem, y0):
    """One folder per labelled section, holding only its pages, legend and check pages."""
    d = os.path.join(VIEWS_DIR, f"{stem}__y{y0}")
    os.makedirs(d, exist_ok=True)
    return d


def save_legend(im, boxes, folder):
    leg = im.crop(boxes["lithology_legend"]).convert("RGB")
    if leg.width > 1600:
        leg = leg.resize((1600, round(leg.height * 1600 / leg.width)))
    path = os.path.join(folder, "legend.png")
    leg.save(path)
    return path


# ---------------------------------------------------------------- draft boundaries

def draft_boundaries(im, box, y0, y1):
    """Draft cuts for rows y0..y1, from the pipeline's own segmentation (lithology.segment).

    Only a starting point: it can still miss very thin beds or add a false cut, and using
    it means labels start from our own guesses, so correct every line rather than
    accepting it. (The old draft measured each row's colour from ALL pixels, so dash
    rows read as colour changes and drew lines all through shale.)"""
    section = (box[0], y0, box[2], y1)
    cuts = detect_boundaries(im.crop(section), edge_margin=section_edge_margins(section, box))
    return [y0 + c for c in cuts[1:-1]]


def predict(img, stem, y0, y1):
    """The full pipeline's intervals for rows y0..y1: segment, cluster, then legend names.
    Returns [{top_y, bottom_y, cluster_id, lithology}], clipped to the section."""
    import tempfile
    from lithology.cluster import run as cluster_run
    from lithology.legend_match import name_clusters
    ann = f"{ANN_DIR}/{stem}.json"
    with tempfile.TemporaryDirectory() as tmp:
        res = cluster_run(img, ann, y0=y0, y1=y1, out_dir=tmp, quiet=True)
    rows = [dict(top_y=max(m["top_y"], y0), bottom_y=min(m["bottom_y"], y1), cluster_id=m["cluster_id"])
            for m in res["merged"] if m["bottom_y"] > y0 and m["top_y"] < y1]
    name_clusters(img, ann, rows)
    return rows


def legend_names(im, boxes):
    """The legend's rock names as the pipeline reads them: type label names exactly like these."""
    from lithology.features import FeatureExtractor
    from lithology.legend_match import legend_swatches
    try:
        return [n for n, _ in legend_swatches(im, boxes["lithology_legend"], FeatureExtractor())]
    except Exception:
        return []


# ---------------------------------------------------------------- label files

def read_labels(path):
    """Parse 'y name' lines (each starts an interval) ending in 'y END'."""
    entries = []
    for n, raw in enumerate(open(path), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            sys.exit(f"{path}:{n}: expected 'y lithology', got '{raw.strip()}'")
        try:
            y = int(parts[0])
        except ValueError:
            sys.exit(f"{path}:{n}: '{parts[0]}' is not a pixel row number")
        entries.append((y, parts[1].strip(), n))
    if not entries or entries[-1][1].upper() != "END":
        sys.exit(f"{path}: the last line must be '<y> END'")
    problems = []
    for (ya, na, la), (yb, _, _) in zip(entries, entries[1:]):
        if yb <= ya:
            problems.append(f"line {la}: rows must increase ({ya} then {yb})")
        if na == "?":
            problems.append(f"line {la}: interval at y={ya} still has lithology '?'")
        if na.upper() == "END":
            problems.append(f"line {la}: END must only be the last line")
    if problems:
        sys.exit("Fix these and run check again:\n  " + "\n  ".join(problems))
    return [(ya, yb, na) for (ya, na, _), (yb, _, _) in zip(entries, entries[1:])]


def colour_for(i):
    h = (i * 0.618034) % 1.0
    return tuple(int(255 * c) for c in colorsys.hsv_to_rgb(h, 0.65, 0.85))


# ---------------------------------------------------------------- commands

def cmd_list(_):
    print(f"{'mode':>4}  {'col_w':>5}  {'col_y0':>7}  {'col_y1':>7}  stem")
    for s in all_stems():
        try:
            _, img, boxes = resolve(s)
        except SystemExit:
            continue
        b = boxes["interpreted_lithology_column"]
        mode = Image.open(img).mode
        print(f"{mode:>4}  {b[2] - b[0]:5d}  {b[1]:7d}  {b[3]:7d}  {s}")


def cmd_show(args):
    stem, img, boxes = resolve(args.stem)
    lith = boxes["interpreted_lithology_column"]
    y0 = max(args.y0, lith[1])
    y1 = min(y0 + args.height, lith[3])
    if y1 - y0 < 50:
        sys.exit(f"Section is outside the column. Column rows are {lith[1]}-{lith[3]}.")
    os.makedirs(LABELS_DIR, exist_ok=True)
    folder = section_dir(stem, y0)
    for old in glob.glob(os.path.join(folder, "page_*.png")):
        os.remove(old)                       # a re-run with a new height leaves no stale pages
    im = Image.open(img)

    txt = label_path(stem, y0, "txt")
    if args.blank:
        draft = [(y, "?") for y in [y0] + draft_boundaries(im, lith, y0, y1)]
    else:
        # Pipeline guess: "unknown" stays "?" so you must fill it; no_data is kept as is.
        draft = [(r["top_y"], "?" if r["lithology"] == "unknown" else r["lithology"])
                 for r in predict(img, stem, y0, y1)]
        draft[0] = (y0, draft[0][1])
    peaks = [y for y, _ in draft[1:]]
    if os.path.exists(txt) and not args.force:
        print(f"Label file exists, NOT overwriting your edits: {txt}\n"
              f"(use --force to replace it with a fresh draft)")
    else:
        names = legend_names(im, boxes)
        with open(txt, "w") as f:
            f.write(f"# {stem}\n# Each line starts an interval: <pixel row> <lithology as named in the legend>\n"
                    f"# Delete a line to merge two intervals; add a line to split one.\n"
                    f"# The last line must be '<row> END'. Rows are full-image pixel rows.\n"
                    f"# Blank paper with no rock drawn: no_data\n")
            if not args.blank:
                f.write("# DRAFT = THE PIPELINE'S GUESS. Check every row and every name; don't just accept it.\n")
            if names:
                f.write("# Legend names (type them exactly like this):\n#   " + ", ".join(names) + "\n")
            for y, n in draft:
                f.write(f"{y} {n}\n")
            f.write(f"{y1} END\n")
        print(f"Draft labels ({len(draft)} intervals): {txt}")

    lines = [(y, str(y), (220, 0, 0)) for y in peaks]
    pages, scale = render_pages(im, boxes, y0, y1, os.path.join(folder, "page"), lines=lines)
    save_legend(im, boxes, folder)
    print(f"Section rows {y0}-{y1}, zoom x{scale}: {len(pages)} page(s) + legend.png in")
    print(f"  {folder}")
    print(f"Look:  open \"{folder}\"")
    print(f"Edit:  open -e \"{txt}\"")


def cmd_check(args):
    stem, y0 = parse_label_name(args.labels)
    stem, img, boxes = resolve(stem)
    lith = boxes["interpreted_lithology_column"]
    intervals = read_labels(args.labels)
    top, bot = intervals[0][0], intervals[-1][1]
    if top < lith[1] or bot > lith[3]:
        sys.exit(f"Rows {top}-{bot} go outside the column ({lith[1]}-{lith[3]}).")

    csv_path = os.path.splitext(args.labels)[0] + ".csv"
    with open(csv_path, "w") as f:
        f.write("top_y,bottom_y,lithology\n")
        for a, b, n in intervals:
            f.write(f"{a},{b},{n}\n")

    names = list(dict.fromkeys(n for _, _, n in intervals))
    cols = {n: colour_for(i) for i, n in enumerate(names)}
    bands = [(a, b, n, cols[n]) for a, b, n in intervals]
    lines = [(a, "", (0, 60, 220)) for a, _, _ in intervals[1:]]
    folder = section_dir(stem, y0)
    for old in glob.glob(os.path.join(folder, "check_*.png")):
        os.remove(old)
    pages, _ = render_pages(Image.open(img), boxes, top, bot, os.path.join(folder, "check"),
                            lines=lines, bands=bands)

    print(f"OK: {len(intervals)} intervals, rows {top}-{bot} -> {csv_path}")
    print(f"{'lithology':24s} {'intervals':>9s} {'rows':>7s} {'>=72px':>7s} {'>=40px':>7s} {'>=24px':>7s}")
    for n in names:
        hs = [b - a for a, b, m in intervals if m == n]
        print(f"{n:24s} {len(hs):9d} {sum(hs):7d} {sum(h >= 72 for h in hs):7d} {sum(h >= 40 for h in hs):7d} {sum(h >= 24 for h in hs):7d}")
    print("(intervals usable at window height 64 / 32 / 16 need >=72 / >=40 / >=24 px)")
    print("Check pages:")
    for p in pages:
        print(f"  {p}")


def cmd_score(args):
    """Score the pipeline against a finished label file."""
    from lithology.evaluate import row_labels, score
    stem, y0 = parse_label_name(args.labels)
    stem, img, boxes = resolve(stem)
    truth = [dict(top_y=a, bottom_y=b, lithology=n) for a, b, n in read_labels(args.labels)]
    top, bot = truth[0]["top_y"], truth[-1]["bottom_y"]
    pred = predict(img, stem, top, bot)
    res = score(truth, pred, len(pred))
    # Name accuracy per row: the share of labelled rows whose predicted name is the label's.
    t = row_labels(truth, "lithology", top, bot)
    p = row_labels(pred, "lithology", top, bot)
    ok = [(a, b) for a, b in zip(t, p) if a is not None and b is not None]
    same = sum(str(a).strip().lower() == str(b).strip().lower() for a, b in ok)
    print(f"{stem}  rows {top}-{bot}: {len(truth)} labelled intervals, {len(pred)} predicted")
    print(f"  boundaries: recall {res['b_recall']:.2f}  precision {res['b_precision']:.2f}  (within 20 px)")
    print(f"  clusters:   row ARI {res['row_ari']:.3f}")
    print(f"  names:      {same / max(1, len(ok)):.1%} of rows named as labelled")
    wrong = {}
    for a, b in ok:
        if str(a).strip().lower() != str(b).strip().lower():
            wrong[(a, b)] = wrong.get((a, b), 0) + 1
    for (a, b), n in sorted(wrong.items(), key=lambda kv: -kv[1])[:8]:
        print(f"    {n:6d} rows: labelled {a!r}, predicted {b!r}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    s = sub.add_parser("show")
    s.add_argument("stem")
    s.add_argument("--y0", type=int, required=True, help="first pixel row of the section")
    s.add_argument("--height", type=int, default=2000, help="section height in pixel rows")
    s.add_argument("--force", action="store_true", help="overwrite an existing label file")
    s.add_argument("--blank", action="store_true", help="draft boundaries only, every name '?'")
    c = sub.add_parser("check")
    c.add_argument("labels")
    sc = sub.add_parser("score")
    sc.add_argument("labels")
    args = ap.parse_args()
    {"list": cmd_list, "show": cmd_show, "check": cmd_check, "score": cmd_score}[args.cmd](args)


if __name__ == "__main__":
    main()
