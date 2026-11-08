"""Run the pipeline on every dataset scan and log crashes, runtime and memory.

    python -m lithology.batch [--timeout SECONDS] [--only SUBSTRING]

Each document runs in its OWN process (python -m lithology.pipeline), for two reasons:
  - a crash, out-of-memory kill or hang loses that document only, not the rest of the run;
  - peak memory is only measurable per process and never goes down, so one process per
    document is the only way to get each document's own peak.
Writes out/batch/batch_log.csv (one row per document) and prints a summary.
"""

import argparse
import glob
import json
import os
import statistics
import subprocess
import sys
import time
from collections import Counter

from .data import DATASET_ANNOTATIONS, DATASET_IMAGES, OUT_DIR
from .output import write_csv

LOG_KEYS = ["document", "status", "error", "seconds", "peak_mb", "n_intervals", "n_clusters",
            "n_no_data", "n_swatches", "n_unknown", "n_absorbed", "n_nan_depths", "n_flagged",
            "qc_light", "qc_reason"]


def run_one(image, annotation, out_dir, timeout, no_boxes=False):
    stem = os.path.splitext(os.path.basename(image))[0]
    t0 = time.perf_counter()
    try:
        cmd = [sys.executable, "-m", "lithology.pipeline", image, annotation, "--out", out_dir]
        p = subprocess.run(cmd + (["--no-boxes"] if no_boxes else []), capture_output=True, text=True,
                           timeout=timeout)
    except subprocess.TimeoutExpired:
        return dict(document=stem, status="timeout", error=f"over {timeout} s",
                    seconds=round(time.perf_counter() - t0, 1))
    if p.returncode == 0:
        return json.loads(p.stdout.strip().splitlines()[-1])   # pipeline's summary line
    # Crashed: keep the last line of the traceback (the exception message). A negative
    # return code means the OS killed it, e.g. -9 is usually out of memory.
    lines = p.stderr.strip().splitlines()
    error = lines[-1] if lines else f"killed by signal {-p.returncode}"
    return dict(document=stem, status="crash", error=error,
                seconds=round(time.perf_counter() - t0, 1))


def summarise(log):
    ok = [r for r in log if r["status"] == "ok"]
    print(f"\n{len(ok)}/{len(log)} documents ok")
    # Same message = same bug. Most frequent first: this is the to-do list.
    errors = Counter(r["error"] for r in log if r.get("error"))
    for msg, n in errors.most_common():
        print(f"  {n:3d} x {msg}")
    # Which documents to trust as they are (lithology.qc): the review list for a human.
    lights = Counter(r.get("qc_light") for r in ok)
    print(f"quality: {lights.get('green', 0)} green, {lights.get('amber', 0)} amber, {lights.get('red', 0)} red")
    for r in ok:
        if r.get("qc_light") in ("amber", "red"):
            print(f"  {r['qc_light']:5s} {r['document'][:45]:45s} {r.get('qc_reason', '')}")
    for key in ("seconds", "peak_mb"):
        v = [r[key] for r in ok if r.get(key) is not None]
        if v:
            print(f"{key}: min {min(v)}  median {statistics.median(v)}  max {max(v)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--timeout", type=int, default=1800, help="per document, seconds")
    ap.add_argument("--only", help="run only scans whose name contains this")
    ap.add_argument("--no-boxes", action="store_true",
                    help="bonus: ignore the supplied boxes, find the regions instead (writes out/batch_no_boxes)")
    a = ap.parse_args()
    out_dir = os.path.join(OUT_DIR, "batch_no_boxes" if a.no_boxes else "batch")   # never mix the two
    os.makedirs(out_dir, exist_ok=True)

    images = sorted(glob.glob(os.path.join(DATASET_IMAGES, "*.tif*")))
    if a.only:
        images = [p for p in images if a.only in os.path.basename(p)]
    log = []
    for i, image in enumerate(images, 1):
        stem = os.path.splitext(os.path.basename(image))[0]
        annotation = os.path.join(DATASET_ANNOTATIONS, stem + ".json")
        if not os.path.exists(annotation):
            r = dict(document=stem, status="crash", error="no annotation JSON")
        else:
            r = run_one(image, annotation, out_dir, a.timeout, a.no_boxes)
        log.append(r)
        print(f"[{i}/{len(images)}] {stem}: {r['status']} {r.get('seconds', '')}s "
              f"{r.get('error') or ''}", flush=True)
        # Rewrite the log after every document, so a long run can be checked while going.
        write_csv(os.path.join(out_dir, "batch_log.csv"),
                  [{k: r.get(k) for k in LOG_KEYS} for r in log], LOG_KEYS)
    summarise(log)


if __name__ == "__main__":
    main()
