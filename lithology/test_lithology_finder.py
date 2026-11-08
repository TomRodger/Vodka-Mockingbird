#!/usr/bin/env python3
"""
Test lithology_finder.py on a folder of images and a folder of annotation files.

Each annotation .json is matched to its image through the "image_name" field inside it (falling
back to the json's own file name), the finder is run with NO access to the boxes, and the result
is scored against the annotated "interpreted_lithology_column" box.

Usage:
  python test_finder.py IMAGES_DIR ANNOTATIONS_DIR [--csv results.csv] [--pass-iou 0.9]
                        [--overlay-dir DIR] [--only substring]

Exit code is 1 if any page fails (no detection or IoU below --pass-iou), so it can run in CI.
"""
import argparse
import csv
import json
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lithology_finder import find_lithology_column, find_by_bands, iou      # noqa: E402

IMG_EXT = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}


def index_images(folder):
    return [p for p in sorted(Path(folder).rglob("*")) if p.suffix.lower() in IMG_EXT]


def _norm(name):
    """Compare names ignoring case, extension and space/underscore/dash differences."""
    return "".join(ch for ch in Path(name).stem.lower() if ch.isalnum())


def match_image(ann_path, ann, images):
    """Find the image for an annotation: exact image_name, then a normalised match of the stem, then
    a name that merely ends with it (e.g. files saved as "<hash>-<name>.tif")."""
    want = ann.get("image_name") or (ann_path.stem + ".tif")
    for p in images:
        if p.name == want:
            return p
    w = _norm(want)
    for p in images:
        if _norm(p.name) == w:
            return p
    for p in images:
        if _norm(p.name).endswith(w):
            return p
    return None


def draw_overlay(image_path, pred, gt, out_png, scale=None):
    from PIL import Image, ImageDraw
    Image.MAX_IMAGE_PIXELS = None
    im = Image.open(image_path).convert("RGB")
    scale = scale or min(1.0, 1600 / im.height)
    im = im.resize((max(1, int(im.width * scale)), max(1, int(im.height * scale))))
    d = ImageDraw.Draw(im)
    d.rectangle([v * scale for v in gt], outline=(0, 170, 0), width=3)            # green = annotation
    if pred:
        d.rectangle([v * scale for v in pred], outline=(255, 0, 0), width=2)      # red = prediction
    im.save(out_png)


def write_diagnostics(image_path, pred, gt, res, out_dir):
    """Small files you can send back for debugging a failed page: a native-resolution crop around the
    annotated column (green = annotation edges, red = predicted edges) and a JSON of what was seen."""
    from PIL import Image, ImageDraw
    Image.MAX_IMAGE_PIXELS = None
    stem = Path(image_path).stem
    im = Image.open(image_path)
    W, H = im.size
    x0, x1 = max(0, gt[0] - 350), min(W, gt[2] + 350)
    mid = (gt[1] + gt[3]) // 2
    y0, y1 = max(0, mid - 700), min(H, mid + 700)
    crop = im.crop((x0, y0, x1, y1)).convert("RGB")
    d = ImageDraw.Draw(crop)
    for x in (gt[0], gt[2]):
        d.line([(x - x0, 0), (x - x0, crop.height)], fill=(0, 170, 0), width=1)
    if pred:
        for x in (pred[0], pred[2]):
            d.line([(x - x0, 0), (x - x0, crop.height)], fill=(255, 0, 0), width=1)
    crop.save(Path(out_dir) / f"{stem}_diag.png")
    info = dict(image=Path(image_path).name, size=[W, H], mode=im.mode, annotation=gt, predicted=pred,
                crop_origin=[x0, y0], mode_used=res.get("mode") if res else None,
                confidence=res.get("confidence") if res else None,
                alternatives=res.get("alternatives") if res else None)
    try:                                                  # what the filled-band method sees on this page
        _, dbg = find_by_bands(image_path, return_debug=True)
        info["band_candidates"] = [{k: round(v, 3) for k, v in c.items()} for c in dbg["candidates"]]
        med = dbg["med"]
        info["fill_profile_near_annotation"] = {int(x): round(float(med[x]), 2)
                                                for x in range(max(0, gt[0] - 40), min(len(med), gt[2] + 40), 2)}
    except Exception as e:
        info["band_debug_error"] = str(e)
    json.dump(info, open(Path(out_dir) / f"{stem}_diag.json", "w"), indent=1, default=str)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images_dir")
    ap.add_argument("annotations_dir")
    ap.add_argument("--csv", help="write per-page results to this CSV")
    ap.add_argument("--pass-iou", type=float, default=0.9, help="IoU needed to count as a pass (default 0.9)")
    ap.add_argument("--overlay-dir", help="write prediction (red) vs annotation (green) overlays here")
    ap.add_argument("--only", help="only run pages whose name contains this text")
    ap.add_argument("--diagnose-dir", help="for pages that fail, write a small crop + JSON here to send for debugging")
    a = ap.parse_args()

    images = index_images(a.images_dir)
    anns = sorted(Path(a.annotations_dir).rglob("*.json"))
    if not images or not anns:
        sys.exit(f"found {len(images)} images and {len(anns)} annotation files - check the folders")
    for d_ in (a.overlay_dir, a.diagnose_dir):
        if d_:
            Path(d_).mkdir(parents=True, exist_ok=True)

    rows, skipped = [], []
    print(f"{'page':44s} {'IoU':>6s} {'conf':>5s} {'mode':>5s} {'sec':>5s}  predicted box vs annotation")
    for ap_ in anns:
        try:
            ann = json.load(open(ap_))
            gt = ann["bounding_boxes"]["interpreted_lithology_column"]
        except Exception as e:
            skipped.append((ap_.name, f"bad annotation: {e}")); continue
        if a.only and a.only not in (ann.get("image_name") or ap_.name):
            continue
        img = match_image(ap_, ann, images)
        if img is None:
            skipped.append((ap_.name, f"no image named {ann.get('image_name')}")); continue

        t = time.time()
        try:
            res = find_lithology_column(img)
            err = ""
        except Exception:
            res, err = None, traceback.format_exc(limit=1).strip().splitlines()[-1]
        sec = time.time() - t
        pred = res["bbox"] if res else None
        score = iou(pred, gt) if pred else 0.0
        rows.append(dict(page=img.name, iou=round(score, 4), found=bool(res), passed=score >= a.pass_iou,
                         confidence=round(res["confidence"], 3) if res else "", mode=res.get("mode", "") if res else "",
                         seconds=round(sec, 2), pred=pred or "", gt=gt, error=err))
        flag = "ok  " if score >= a.pass_iou else ("MISS" if not res else "LOW ")
        print(f"{flag}{img.name[:40]:40s} {score:6.3f} {rows[-1]['confidence'] or 0:5} {rows[-1]['mode']:>5s} {sec:5.1f}  "
              f"{pred} vs {gt} {err}")
        if a.overlay_dir:
            draw_overlay(img, pred, gt, Path(a.overlay_dir) / (img.stem + "_check.png"))
        if a.diagnose_dir and score < a.pass_iou:
            try:
                write_diagnostics(img, pred, gt, res, a.diagnose_dir)
            except Exception as e:
                print(f"     (could not write diagnostics: {e})")

    for name, why in skipped:
        print(f"SKIP {name}: {why}")
    if not rows:
        sys.exit("nothing was scored")

    ious = [r["iou"] for r in rows]
    print("\n" + "=" * 70)
    print(f"pages scored : {len(rows)}   (skipped {len(skipped)})")
    print(f"detected     : {sum(r['found'] for r in rows)}/{len(rows)}")
    print(f"pass (IoU>={a.pass_iou}): {sum(r['passed'] for r in rows)}/{len(rows)}")
    print(f"mean IoU     : {sum(ious) / len(ious):.3f}   min {min(ious):.3f}   median {sorted(ious)[len(ious) // 2]:.3f}")
    print(f"time / page  : mean {sum(r['seconds'] for r in rows) / len(rows):.1f}s   max {max(r['seconds'] for r in rows):.1f}s")
    worst = sorted(rows, key=lambda r: r["iou"])[:3]
    print("worst pages  : " + ", ".join(f"{r['page'][:30]} ({r['iou']:.3f})" for r in worst))

    if a.csv:
        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"wrote {a.csv}")
    sys.exit(0 if all(r["passed"] for r in rows) else 1)


if __name__ == "__main__":
    main()