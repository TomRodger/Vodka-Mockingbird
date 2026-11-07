"""Read the scan's printed depth ruler so layer positions can become depths.

Input: the original TIFF scan and its supplied annotation JSON.
Output: markers linking full-image pixel rows to printed depth numbers.
OCR means reading text from an image. It does not find rock layers here.
"""

import argparse
import csv
import json
import re
import sys
from itertools import combinations
from bisect import bisect_left
from statistics import median
from pathlib import Path

# Look for dependencies installed in this folder. Normal pip installs work too.
local_packages = Path(__file__).parent / ".python-packages"
if local_packages.is_dir():
    sys.path.insert(0, str(local_packages))


def parse_depth(text):
    """Turn a number such as '25,400' into 25400; ignore non-numeric text."""
    text = text.strip().replace(" ", "")
    if re.fullmatch(r"\d{1,3}(,\d{3})+(\.\d+)?", text):
        text = text.replace(",", "")
    if not re.fullmatch(r"\d+(\.\d+)?", text):
        return None
    value = float(text)
    # Joined labels such as '177001780017900' are not usable depth readings.
    return value if value <= 1_000_000 else None


def _select_uniform_markers(candidates, tolerance=25):
    """Remove readings that do not fit the same steadily increasing ruler.

    OCR can misread digits. We try pairs of labels and keep the largest group
    agreeing with one scale. tolerance is the allowed position error in pixels.
    """
    candidates = sorted(candidates, key=lambda r: r["pixel_y"])
    # Chunks overlap, so one label may be read twice. Merge nearby readings
    # with the same depth; keep conflicting numbers for the ruler to resolve.
    unique = []
    for row in sorted(candidates, key=lambda r: -r["confidence"]):
        if not any(row["depth"] == other["depth"] and abs(row["pixel_y"] - other["pixel_y"]) <= 50
                   for other in unique):
            unique.append(row)
    unique.sort(key=lambda r: r["pixel_y"])
    # Keep conflicting readings of the same label until the ruler checks them.
    # OCR confidence alone cannot tell '10610' from a misread '0610'. Count
    # physical label positions, so two readings do not count as two votes.
    support_count = 0
    anchor = float("-inf")
    for row in unique:
        if row["pixel_y"] - anchor >= 10:
            support_count += 1
            anchor = row["pixel_y"]
    best = []
    # On long scans, try pairs from a smaller sample to save time. Still check
    # each possible scale against every reading.
    step = max(1, len(unique) // 80)
    sample = unique[::step]
    for a, b in combinations(sample, 2):
        dy = b["pixel_y"] - a["pixel_y"]
        dd = b["depth"] - a["depth"]
        if dy <= 0 or dd <= 0:
            continue
        slope = dd / dy  # How much depth changes for each pixel down the image.
        matches = [r for r in unique if abs(
            (r["depth"] - a["depth"]) / slope + a["pixel_y"] - r["pixel_y"]
        ) <= tolerance]
        if len(matches) > len(best):
            best = matches
            best_scale = (a["pixel_y"], a["depth"], slope)
    # Require at least three agreeing labels and support from 60% of positions.
    if len(best) < 3 or len(best) < 0.6 * support_count:
        raise ValueError("No reliable depth scale: need at least 3 agreeing labels and 60% agreement. Inspect ocr_candidates.csv and depth_preview.png.")
    groups = []
    for row in best:
        if not groups or row["pixel_y"] - groups[-1][0]["pixel_y"] >= 10:
            groups.append([])
        groups[-1].append(row)
    reference_y, reference_depth, slope = best_scale
    # Where several readings describe the same label, keep the one closest
    # to the ruler we found, rather than automatically trusting OCR confidence.
    best = [min(group, key=lambda r: abs((r["depth"] - reference_depth) / slope + reference_y - r["pixel_y"]))
            for group in groups]
    # A repeated or slightly misplaced reading can still sit near the fitted
    # ruler. Keep the longest strictly increasing sequence, without inventing
    # or changing any depth numbers.
    tails, tail_indices, previous = [], [], []
    # Track the best increasing sequence efficiently. previous stores links
    # so we can recover the original readings after checking the whole list.
    for index, row in enumerate(best):
        slot = bisect_left(tails, row["depth"])
        previous.append(tail_indices[slot - 1] if slot else -1)
        if slot == len(tails):
            tails.append(row["depth"])
            tail_indices.append(index)
        else:
            tails[slot] = row["depth"]
            tail_indices[slot] = index
    kept = []
    index = tail_indices[-1]
    while index != -1:
        kept.append(best[index])
        index = previous[index]
    kept.reverse()
    if len(kept) < 3 or len(kept) < 0.6 * support_count:
        raise ValueError("Too few consistent readings after removing repeated or reversed labels.")
    return kept


def select_markers(candidates, tolerance=25):
    """Check the whole ruler first, then allow small changes in scan stretch.

    Some long scans stretch slightly from one section to the next. Check short
    sections separately in that case, while still requiring 60% support and
    similar depth-per-pixel scales. Abrupt scale breaks remain an error.
    """
    # Some rulers print 1500, then abbreviated 20/40/60/80, then 1600.
    # Use the full anchor labels rather than pretending those small numbers
    # mean shallow depths. Require repeated small values and a verified full
    # ruler before excluding them; genuine shallow-depth labels stay untouched.
    full = [r for r in candidates if r["depth"] >= 1000]
    small_positions = {}
    for row in candidates:
        if row["depth"] < 100:
            small_positions.setdefault(row["depth"], []).append(row["pixel_y"])
    repeated_small = sum(len(positions) >= 3 and max(positions) - min(positions) > 100
                         for positions in small_positions.values())
    if len(full) >= 3 and repeated_small >= 2:
        try:
            _select_uniform_markers(full, tolerance)
        except ValueError:
            pass
        else:
            candidates = [r for r in candidates if r["depth"] >= 100]
    try:
        return _select_uniform_markers(candidates, tolerance)
    except ValueError as original_error:
        ordered = sorted(candidates, key=lambda r: r["pixel_y"])
        if len(ordered) <= 100:
            raise
        sections = []
        kept = []
        for start in range(0, len(ordered), 100):
            section = ordered[start:start + 100]
            try:
                accepted = _select_uniform_markers(section, tolerance)
            except ValueError:
                continue
            sections.append(median((b["depth"] - a["depth"]) / (b["pixel_y"] - a["pixel_y"])
                                   for a, b in zip(accepted, accepted[1:])))
            # A repeated label on a section boundary belongs in the ruler once.
            for row in accepted:
                if not kept or row["depth"] > kept[-1]["depth"] and row["pixel_y"] > kept[-1]["pixel_y"]:
                    kept.append(row)
        # Compare support with distinct labels, excluding overlap duplicates.
        distinct = 0
        anchor = float("-inf")
        for row in ordered:
            if row["pixel_y"] - anchor >= 10:
                distinct += 1
                anchor = row["pixel_y"]
        if len(kept) < 3 or len(kept) < 0.6 * distinct or len(sections) < 2:
            raise original_error
        typical_scale = median(sections)
        # Neighbouring sections must still join into one believable ruler.
        # A change greater than 10% needs review instead of automatic conversion.
        joins = [(b["depth"] - a["depth"]) / (b["pixel_y"] - a["pixel_y"])
                 for a, b in zip(kept, kept[1:])]
        if any(abs(scale / typical_scale - 1) > 0.1 for scale in sections + joins):
            raise ValueError("Ruler has an abrupt scale change or inconsistent sections.")
        return kept


def write_csv(path, rows, fields):
    """Save readings in a table that the converter can open later."""
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def create_ocr_engine(detail=False):
    """Use small ruler crops, or enlarged input for an optional comparison."""
    from rapidocr import RapidOCR

    settings = {
        "EngineConfig.onnxruntime.intra_op_num_threads": 1,
        "EngineConfig.onnxruntime.inter_op_num_threads": 1,
    }
    if not detail:
        settings.update({"Det.limit_type": "max", "Det.limit_side_len": 1200})
    return RapidOCR(params=settings)


def read_ruler(image, box, engine, chunk_height, rotation=0):
    """Read small crops, then map rotated OCR positions back to the full image."""
    import numpy as np
    from PIL import Image

    x0, y0, x1, y1 = box
    readings = []
    if chunk_height <= 120:
        raise ValueError("OCR crop height must be greater than 120 pixels.")
    starts = list(range(y0, y1, chunk_height - 120))
    tile_width = chunk_height + 40 if rotation else x1 - x0 + 80
    # Short fallback crops can share one OCR call. Blank margins keep the
    # sections separate, and each section retains its original image position.
    batch_size = max(1, 1200 // (tile_width + 20)) if chunk_height <= 240 else 1
    for group in range(0, len(starts), batch_size):
        tiles = []
        offset = 0
        for start in starts[group:group + batch_size]:
            end = min(y1, start + chunk_height)
            crop = image.crop((x0, start, x1, end)).convert("RGB")
            padded = Image.new("RGB", (crop.width + 80, crop.height + 40), "white")
            padded.paste(crop, (40, 20))
            rotated = padded.rotate(rotation, expand=True) if rotation else padded
            tiles.append((start, padded, rotated, offset))
            offset += rotated.width + 20
        sheet = Image.new("RGB", (offset - 20, max(t[2].height for t in tiles)), "white")
        for _, _, rotated, left in tiles:
            sheet.paste(rotated, (left, 0))
        result = engine(np.asarray(sheet), use_cls=False)
        if result.txts is not None:
            for points, text, score in zip(result.boxes, result.txts, result.scores):
                depth = parse_depth(text)
                if depth is None or score < 0.7:
                    continue
                points = np.asarray(points, dtype=float)
                centre_x = float(points[:, 0].mean())
                tile = next((t for t in tiles if t[3] <= centre_x < t[3] + t[2].width), None)
                if tile is None:
                    continue
                start, padded, rotated, left = tile
                points = points.copy()
                # OCR coordinates start on the combined sheet. First move
                # them back into the individual crop, then undo its rotation.
                points[:, 0] -= left
                if points[:, 0].min() < 0 or points[:, 0].max() > rotated.width:
                    continue  # Never read a number spanning two separate crops.
                original = np.asarray(padded)
                if rotation == 90:
                    ox, oy = padded.width - points[:, 1], points[:, 0]
                elif rotation == 270:
                    ox, oy = points[:, 1], padded.height - points[:, 0]
                else:
                    ox, oy = points[:, 0], points[:, 1]
                # Use the label centre, remove the white top margin, then add
                # the crop's starting row to get the original full-image row.
                y = float(np.mean(oy)) - 20 + start
                if not y0 <= y <= y1:
                    continue
                # Preserve the distinction between black ruler labels and a
                # coloured secondary scale. Do not mix both sets of numbers.
                patch = original[max(0, int(oy.min())):min(padded.height, int(oy.max()) + 1),
                                 max(0, int(ox.min())):min(padded.width, int(ox.max()) + 1)]
                ink = patch.min(axis=2) < 180
                coloured = patch.max(axis=2).astype(int) - patch.min(axis=2).astype(int) > 45
                colour_fraction = float((ink & coloured).sum()) / max(1, int(ink.sum()))
                readings.append({"pixel_y": round(y, 2), "depth": depth,
                                 "confidence": round(float(score), 4), "text": text,
                                 "ink": "colour" if colour_fraction > 0.35 else "black"})
        first_row = tiles[0][0]
        last_row = min(y1, tiles[-1][0] + chunk_height)
        print(f"Rotation {rotation}: rows {first_row}-{last_row}, {len(readings)} numeric readings", flush=True)
    return readings


def ruler_series(readings, series="primary"):
    """Use the main black ruler when a coloured secondary ruler is present.

    If the only ruler is coloured, keep it. A caller can explicitly request the
    coloured secondary series; this does not by itself identify MD versus TVD.
    """
    if series == "all":
        return readings
    black = [r for r in readings if r.get("ink") == "black"]
    colour = [r for r in readings if r.get("ink") == "colour"]
    if series == "secondary":
        if len(colour) < 3:
            raise ValueError("No coloured secondary ruler was found.")
        return colour
    if len(black) >= 3 and len(colour) >= 3 and len(black) >= 0.25 * len(readings):
        return black
    return readings


def extract(image_path, annotation_path, output_dir, chunk_height=1200, engine=None, series="primary"):
    """Read one scan's ruler and return its accepted depth markers.

    Paths should be pathlib.Path objects. output_dir must be a new folder.
    An existing OCR engine can be passed in when processing multiple scans.
    Returns a list of dictionaries: pixel_y is the full-image row, and depth
    is the printed number. Pass those two fields to DepthConverter.
    series selects the main ruler or the coloured secondary ruler, not a unit.
    """
    from PIL import Image, ImageDraw

    # The supplied JSON tells us where the depth ruler sits in the full image.
    annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
    x0, y0, x1, y1 = annotation["bounding_boxes"]["depth_column"]
    # Challenge scans are unusually tall, so allow their known large sizes.
    Image.MAX_IMAGE_PIXELS = None
    output_dir.mkdir(parents=True, exist_ok=False)
    candidates = []
    engine = engine if engine is not None else create_ocr_engine()
    with Image.open(image_path) as image:
        if image.size != (annotation["width"], annotation["height"]):
            raise ValueError("Image dimensions do not match its annotation.")
        if not (0 <= x0 < x1 <= image.width and 0 <= y0 < y1 <= image.height):
            raise ValueError("Depth-column bounding box is outside the image.")
        preview_bottom = min(y1, y0 + 1800)
        preview = image.crop((x0, y0, x1, preview_bottom)).convert("RGB")
        preview.save(output_dir / "depth_preview.png")
        fields = ["pixel_y", "depth", "confidence", "text", "ink"]
        attempts = []
        # Try the usual orientation first. Rotate only when it fails, so normal
        # scans stay fast. Every orientation is mapped back to original rows.
        # Shorter crops are a final fallback: they stop neighbouring labels and
        # grid lines from being mistaken for part of one long text block.
        configurations = [(r, chunk_height) for r in (0, 90, 270)]
        if chunk_height > 240:
            configurations += [(r, 240) for r in (0, 90, 270)]
        for rotation, read_height in configurations:
            candidates = read_ruler(image, (x0, y0, x1, y1), engine, read_height, rotation)
            write_csv(output_dir / f"ocr_rotation_{rotation}_height_{read_height}.csv", candidates, fields)
            write_csv(output_dir / "ocr_candidates.csv", candidates, fields)
            try:
                selected = ruler_series(candidates, series)
                markers = select_markers(selected)
                attempts.append({"rotation": rotation, "chunk_height": read_height, "readings": len(candidates), "selected_series_readings": len(selected), "status": "accepted"})
                break
            except ValueError as error:
                attempts.append({"rotation": rotation, "chunk_height": read_height, "status": "failed", "error": str(error)})
        else:
            (output_dir / "diagnostics.json").write_text(json.dumps({"series": series, "attempts": attempts}, indent=2))
            raise ValueError("No reliable ruler after normal and rotated OCR. Inspect diagnostics.json.")
        (output_dir / "diagnostics.json").write_text(json.dumps({"series": series, "rotation": rotation, "chunk_height": read_height, "attempts": attempts}, indent=2))
    write_csv(output_dir / "markers.csv", markers, fields)
    # Draw accepted labels on the first section so a person can check them.
    canvas = Image.new("RGB", (preview.width + 200, preview.height), "white")
    canvas.paste(preview, (0, 0))
    draw = ImageDraw.Draw(canvas)
    for row in markers:
        y = row["pixel_y"] - y0
        if 0 <= y < canvas.height:
            draw.line((0, y, preview.width, y), fill="red", width=1)
            draw.text((preview.width + 8, y), f"{row['depth']:g} @ {row['pixel_y']:g}", fill="black")
    canvas.save(output_dir / "depth_preview.png")
    print(f"Accepted {len(markers)} markers. Depth range: {markers[0]['depth']:g}-{markers[-1]['depth']:g}.")
    print(f"Saved markers.csv and depth_preview.png in {output_dir}")
    return markers


def main():
    """Allow extraction to be run from the terminal with image/JSON paths."""
    project = Path(__file__).resolve().parents[2] / "hack27" / "project"   # organisers' material
    default_image = project / "example/0000000009_d-582_0000020435-0001_log06_fe_5in_md.tif"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=default_image)
    parser.add_argument("--annotation", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("results/example"))
    parser.add_argument("--series", choices=("primary", "secondary", "all"), default="primary",
                        help="Main black ruler, coloured secondary ruler, or all readings")
    args = parser.parse_args()
    try:
        extract(args.image, args.annotation or args.image.with_suffix(".json"), args.output_dir, series=args.series)
    except (ValueError, KeyError, OSError, ImportError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
