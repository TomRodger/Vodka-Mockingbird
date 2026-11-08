"""Run the whole pipeline on ONE document: segment -> cluster -> legend -> depth.

    python -m lithology.pipeline IMAGE ANNOTATION_JSON [--out DIR]

Writes <DIR>/<scan>.csv (the brief's output) and <DIR>/<scan>_annotated.png, and prints
one JSON line at the end with timing, peak memory and counts. lithology.batch reads that
line to build its log, so keep it the LAST thing printed.

Lithology names come from lithology.legend_match: "unknown" when no legend swatch is close.
"""

import argparse
import json
import math
import os
import resource
import shutil
import sys
import time
from pathlib import Path

from .cluster import run as cluster_run
from .data import OUT_DIR
from .depth.depth_converter import DepthConverter
from .depth.extract_depths import extract as extract_markers
from .legend_match import name_clusters
from .output import save_annotated, write_csv

# The organisers' example.csv says bottom_depth; the slides say base_depth. Write both
# (same values) until they confirm: a scorer ignores a column it doesn't ask for, but
# fails on one that's missing.
CSV_KEYS = ["top_y", "bottom_y", "cluster_id", "lithology", "top_depth", "bottom_depth", "base_depth"]


def add_depths(rows, image, annotation, work_dir):
    """Read the ruler and add depths. If the ruler can't be read, the depths are NaN and the
    document still gets a CSV; the error is returned so the batch log shows it."""
    depth_dir = Path(work_dir) / "depth"
    shutil.rmtree(depth_dir, ignore_errors=True)       # extract() needs a fresh folder
    try:
        markers = extract_markers(Path(image), Path(annotation), depth_dir)
        converter = DepthConverter((m["pixel_y"], m["depth"]) for m in markers)
        error = None
    except Exception as e:                             # OCR failures raise ValueError, but
        converter, error = None, f"{type(e).__name__}: {e}"   # one bad ruler mustn't kill the run
    out = []
    for r in rows:
        r = dict(r, bottom_depth=None)                 # converter fills bottom_depth if present
        if converter:
            r = converter.convert_interval(r)
        else:
            r.update(top_depth=math.nan, base_depth=math.nan, bottom_depth=math.nan)
        out.append(r)
    return out, error


def peak_memory_mb():
    """Peak memory of THIS process so far. ru_maxrss is bytes on macOS, kilobytes on Linux."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(peak / (1024 ** 2 if sys.platform == "darwin" else 1024), 1)


def run(image, annotation, out_dir=None):
    t0 = time.perf_counter()
    stem = os.path.splitext(os.path.basename(image))[0]
    out_dir = out_dir or os.path.join(OUT_DIR, "pipeline")
    work_dir = os.path.join(out_dir, stem + "_work")   # intermediate files
    os.makedirs(work_dir, exist_ok=True)

    res = cluster_run(image, annotation, out_dir=work_dir, quiet=True)   # segments too
    rows = [dict(top_y=m["top_y"], bottom_y=m["bottom_y"], cluster_id=m["cluster_id"])
            for m in res["merged"]]
    n_swatches = name_clusters(image, annotation, rows)
    rows, depth_error = add_depths(rows, image, annotation, work_dir)

    write_csv(os.path.join(out_dir, stem + ".csv"), rows, CSV_KEYS)
    save_annotated(os.path.join(out_dir, stem + "_annotated.png"), image, annotation, rows)

    return dict(document=stem, status="ok", error=depth_error,
                seconds=round(time.perf_counter() - t0, 1), peak_mb=peak_memory_mb(),
                n_intervals=len(rows), n_clusters=res["n_clusters"],
                n_no_data=sum(r["cluster_id"] < 0 for r in rows), n_swatches=n_swatches,
                n_unknown=sum(r["lithology"] == "unknown" for r in rows),
                n_nan_depths=sum(math.isnan(r[k]) for r in rows for k in ("top_depth", "base_depth")))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image")
    ap.add_argument("annotation")
    ap.add_argument("--out", help="output folder (default: out/pipeline)")
    a = ap.parse_args()
    print(json.dumps(run(a.image, a.annotation, a.out)))


if __name__ == "__main__":
    main()
