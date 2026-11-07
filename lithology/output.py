"""Files the pipeline writes: CSVs and images for checking results by eye."""

import colorsys
import csv

import numpy as np
from PIL import Image, ImageDraw, ImageFont

Image.MAX_IMAGE_PIXELS = None


def font(size):
    for name in ("DejaVuSans.ttf", "Arial.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def palette(k):
    """k distinct colours (golden-ratio hues, so neighbours stay distinct)."""
    return [tuple(int(255 * c) for c in colorsys.hsv_to_rgb((i * 0.618034) % 1, 0.7, 0.9))
            for i in range(k)]


def write_csv(path, rows, keys):
    """Rows of dicts to CSV. Set values (flags) are written as 'a|b|c'."""
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("|".join(sorted(r[k])) if isinstance(r[k], set) else r[k]) for k in keys})


def draw_bounding_boxes(image_path, bounding_boxes, output_path="annotated.png", max_height=4000):
    """Draw the annotation boxes on a shrunken overview of the whole page.

    The page is converted to RGB a strip at a time and shrunk as it goes, so a
    300,000-px-tall log never exists at full size as RGB (that would be ~1.6 GB)."""
    image = Image.open(image_path)
    scale = min(1.0, max_height / image.height)
    out = Image.new("RGB", (max(1, round(image.width * scale)), max(1, round(image.height * scale))))
    strip = 4000
    for ys in range(0, image.height, strip):
        ye = min(ys + strip, image.height)
        part = image.crop((0, ys, image.width, ye)).convert("RGB")
        top, bottom = round(ys * scale), round(ye * scale)
        out.paste(part.resize((out.width, max(1, bottom - top))), (0, top))
    draw = ImageDraw.Draw(out)
    for name, (x1, y1, x2, y2) in bounding_boxes.items():
        draw.rectangle((x1 * scale, y1 * scale, x2 * scale, y2 * scale), outline="red", width=3)
    out.save(output_path)
    print(f"Saved annotated overview to {output_path}")


def save_overlay(path, col, col_y0, merged, k, max_h=6000):
    """Lithology column with a cluster-coloured bar and boundary lines (scaled to fit).
    col is the column as an RGB array starting at full-image row col_y0."""
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
            d.rectangle([x - 1, y - 1, x + tw, y + th],
                        outline=(220, 0, 0) if descs[i]["thin"] else (160, 160, 160))
    img.save(path)
