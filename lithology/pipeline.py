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
import shutil
import sys
import time
from pathlib import Path

try:
    import resource
except ImportError:  # Windows has no resource module.
    resource = None

from .cluster import run as cluster_run
from .data import OUT_DIR, load_annotation
from .depth.depth_converter import DepthConverter
from .depth.extract_depths import extract as extract_markers
from .inclusions import absorb_inclusions
from . import qc
from .legend_match import name_clusters
from .output import save_annotated, write_csv

# The organisers' example.csv says bottom_depth; the slides say base_depth. Write both
# (same values) until they confirm: a scorer ignores a column it doesn't ask for, but
# fails on one that's missing.
CSV_KEYS = ["top_y", "bottom_y", "cluster_id", "lithology", "top_depth", "bottom_depth", "base_depth"]
# <scan>_qc.csv: only the intervals with a flag, and why (lithology.qc). The brief's CSV is unchanged.
QC_KEYS = ["top_y", "bottom_y", "lithology", "top_depth", "base_depth", "flags"]


def add_depths(rows, image, annotation, work_dir, auto_depth=False, depth_candidate=None):
    """Read the ruler and add depths. If the ruler can't be read, the depths are NaN and the
    document still gets a CSV; the error is returned so the batch log shows it."""
    depth_dir = Path(work_dir) / "depth"
    shutil.rmtree(depth_dir, ignore_errors=True)       # extract() needs a fresh folder
    try:
        markers = extract_markers(Path(image), Path(annotation), depth_dir,
                                  auto_depth=auto_depth, depth_candidate=depth_candidate)
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
    """Peak memory of this process, on Windows, macOS or Linux."""
    if resource is None:
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                (name, ctypes.c_size_t) for name in (
                    "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                    "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
                    "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            raise ctypes.WinError(ctypes.get_last_error())
        return round(counters.PeakWorkingSetSize / 1024 ** 2, 1)
    # ru_maxrss is bytes on macOS and kilobytes on Linux.
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(peak / (1024 ** 2 if sys.platform == "darwin" else 1024), 1)


def run(image, annotation, out_dir=None, auto_depth=False, depth_candidate=None, no_boxes=False):
    t0 = time.perf_counter()
    stem = os.path.splitext(os.path.basename(image))[0]
    out_dir = out_dir or os.path.join(OUT_DIR, "pipeline")
    work_dir = os.path.join(out_dir, stem + "_work")   # intermediate files
    os.makedirs(work_dir, exist_ok=True)
    if no_boxes:
        # Bonus: find the three regions ourselves and run the same pipeline on them.
        from .find_boxes import annotation_for
        annotation = annotation_for(image, os.path.join(work_dir, "found_boxes.json"))

    res = cluster_run(image, annotation, out_dir=work_dir, quiet=True)   # segments too
    rows = [dict(top_y=m["top_y"], bottom_y=m["bottom_y"], cluster_id=m["cluster_id"])
            for m in res["merged"]]
    swatch_names = name_clusters(image, annotation, rows)
    n_swatches = len(swatch_names)
    rows, n_absorbed = absorb_inclusions(image, load_annotation(annotation)["interpreted_lithology_column"], rows)
    rows, depth_error = add_depths(rows, image, annotation, work_dir, auto_depth, depth_candidate)

    write_csv(os.path.join(out_dir, stem + ".csv"), rows, CSV_KEYS)
    # Quality checks (lithology.qc): which intervals to check, and a light for the document.
    for r in rows:
        r["flags"] = "|".join(qc.flags(r))
    write_csv(os.path.join(out_dir, stem + "_qc.csv"), [r for r in rows if r["flags"]], QC_KEYS)
    try:
        missing, merged = qc.legend_coverage(image, annotation, swatch_names)
    except Exception:
        missing, merged = [], []
    light, why = qc.document_light(rows, n_swatches, missing, merged)
    save_annotated(os.path.join(out_dir, stem + "_annotated.png"), image, annotation, rows)

    return dict(document=stem, status="ok", error=depth_error,
                seconds=round(time.perf_counter() - t0, 1), peak_mb=peak_memory_mb(),
                n_intervals=len(rows), n_clusters=res["n_clusters"],
                n_no_data=sum(r["cluster_id"] < 0 for r in rows), n_swatches=n_swatches,
                n_unknown=sum(r["lithology"] == "unknown" for r in rows), n_absorbed=n_absorbed,
                n_nan_depths=sum(math.isnan(r[k]) for r in rows for k in ("top_depth", "base_depth")),
                qc_light=light, qc_reason=why, n_flagged=sum(bool(r["flags"]) for r in rows))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image")
    ap.add_argument("annotation", nargs="?", help="supplied boxes (not needed with --no-boxes)")
    ap.add_argument("--out", help="output folder (default: out/pipeline)")
    ap.add_argument("--auto-depth", action="store_true", help="Find the depth column without its annotation box")
    ap.add_argument("--depth-candidate", type=int, help="Select an automatic ruler candidate (starting at 0)")
    ap.add_argument("--no-boxes", action="store_true",
                    help="bonus: find the column, legend and ruler without the supplied boxes")
    a = ap.parse_args()
    if not a.annotation and not a.no_boxes:
        ap.error("give ANNOTATION, or use --no-boxes")
    if a.depth_candidate is not None and not a.auto_depth:
        ap.error("Use --auto-depth with --depth-candidate.")
    print(json.dumps(run(a.image, a.annotation, a.out, a.auto_depth, a.depth_candidate, a.no_boxes)))


if __name__ == "__main__":
    main()
