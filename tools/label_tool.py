"""
Label sections of an interpreted lithology column quickly.

Run from the repo root:

  python -m tools.label_tool list
      every annotated document, with its column size

  python -m tools.label_tool show STEM --y0 10335 --height 2000
      renders the section (zoomed, with a pixel ruler) and the legend into
      out/views/, and writes a DRAFT label file
      tools/labels/STEM__y10335.txt with guessed boundaries

  python -m tools.label_tool check tools/labels/STEM__y10335.txt
      validates your edited label file, writes the matching .csv, and renders
      your labels over the section so you can verify them

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
from scipy.ndimage import median_filter

Image.MAX_IMAGE_PIXELS = None

from lithology.data import DATASET_ANNOTATIONS as ANN_DIR, DATASET_IMAGES as IMAGES_DIR, OUT_DIR

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


def save_legend(im, boxes, stem):
    leg = im.crop(boxes["lithology_legend"]).convert("RGB")
    if leg.width > 1600:
        leg = leg.resize((1600, round(leg.height * 1600 / leg.width)))
    path = os.path.join(VIEWS_DIR, f"{stem}__legend.png")
    leg.save(path)
    return path


# ---------------------------------------------------------------- draft boundaries

def draft_boundaries(im, box, y0, y1):
    """Guess boundaries from changes in FILL COLOUR only. Only a draft:
    it cannot see texture-only changes (so it finds nothing in black-and-white
    documents) and may add a few false lines. Check every line."""
    a = np.asarray(im.crop((box[0], y0, box[2], y1)).convert("RGB"), dtype=float) / 255
    H, W = a.shape[:2]
    t = max(1, W // 8)
    inner = a[:, t:W - t]
    sig = median_filter(np.median(inner, axis=1), size=(7, 1))   # fill colour per row
    k = 4
    score = np.zeros(H)
    score[k:H - k] = np.abs(sig[2 * k:] - sig[:-2 * k]).sum(axis=1) / 0.25
    peaks = []
    for y in np.argsort(-score):
        if score[y] < 1:
            break
        if all(abs(y - p) > 8 for p in peaks):
            peaks.append(int(y))
    return sorted(y0 + p for p in peaks if 6 < p < H - 6)


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
    os.makedirs(VIEWS_DIR, exist_ok=True)
    im = Image.open(img)

    txt = label_path(stem, y0, "txt")
    if im.mode == "1":
        peaks = []
        print("Black-and-white image: no colour to draft from, so the draft is one interval. "
              "Add a line for every boundary yourself, reading rows off the ruler.")
    else:
        peaks = draft_boundaries(im, lith, y0, y1)
    if os.path.exists(txt) and not args.force:
        print(f"Label file exists, NOT overwriting your edits: {txt}\n"
              f"(use --force to replace it with a fresh draft)")
    else:
        with open(txt, "w") as f:
            f.write(f"# {stem}\n# Each line starts an interval: <pixel row> <lithology as named in the legend>\n"
                    f"# Delete a line to merge two intervals; add a line to split one.\n"
                    f"# The last line must be '<row> END'. Rows are full-image pixel rows.\n")
            for y in [y0] + peaks:
                f.write(f"{y} ?\n")
            f.write(f"{y1} END\n")
        print(f"Draft labels ({len(peaks) + 1} intervals): {txt}")

    lines = [(y, str(y), (220, 0, 0)) for y in peaks]
    pages, scale = render_pages(im, boxes, y0, y1,
                                os.path.join(VIEWS_DIR, f"{stem}__y{y0}"), lines=lines)
    print(f"Legend: {save_legend(im, boxes, stem)}")
    print(f"Section rows {y0}-{y1}, zoom x{scale}, {len(pages)} page(s):")
    for p in pages:
        print(f"  {p}")


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
    os.makedirs(VIEWS_DIR, exist_ok=True)
    pages, _ = render_pages(Image.open(img), boxes, top, bot,
                            os.path.join(VIEWS_DIR, f"{stem}__y{y0}_check"),
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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    s = sub.add_parser("show")
    s.add_argument("stem")
    s.add_argument("--y0", type=int, required=True, help="first pixel row of the section")
    s.add_argument("--height", type=int, default=2000, help="section height in pixel rows")
    s.add_argument("--force", action="store_true", help="overwrite an existing label file")
    c = sub.add_parser("check")
    c.add_argument("labels")
    args = ap.parse_args()
    {"list": cmd_list, "show": cmd_show, "check": cmd_check}[args.cmd](args)


if __name__ == "__main__":
    main()
