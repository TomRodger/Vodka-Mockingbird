# Vodka Mockingbird

Viridien challenge (HackTheBurgh): from scanned well logs, automatically extract the
lithology intervals, group them by visual pattern, name each from the log's own legend and
give their depths. Every run also says which outputs to trust.

## Setup (from the repo root)

```
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
brew install tesseract          # Ubuntu: apt install tesseract-ocr. Needed for legend names.
python -m pytest tests/         # should all pass
```

Run modules as `python -m lithology.<module>` from the repo root, not as files: they import
each other as a package.

## Running it

```
python -m lithology.pipeline IMAGE ANNOTATION_JSON     # one document
python -m lithology.batch                              # every dataset scan
```

Per document, in `out/pipeline/` (or `out/batch/`, git-ignored):

| file | what it is |
|---|---|
| `<scan>.csv` | the brief's output: `top_y, bottom_y, cluster_id, lithology, top_depth, bottom_depth, base_depth` |
| `<scan>_annotated.png` | the original scan with boundaries, names and depths drawn on (tall logs: `_p01`, `_p02`, ... pages) |
| `<scan>_qc.csv` | only the intervals worth checking, with the reason |

`batch` also writes `batch_log.csv` (status, runtime, peak memory, counts and a
green / amber / red quality light per document) and ends with the amber/red review list.

## Layout

```
lithology/            the pipeline (details: lithology/README.md)
  segment.py            1. interval boundaries in the lithology column
  cluster.py            2. group intervals by visual pattern
  legend_extractor.py   3a. cut the legend into named swatches
  legend_match.py       3b. name each group from its own document's legend
  inclusions.py         merge small partial-width symbols into the rock around them
  depth/                4. read the depth ruler, convert pixel rows to depths
  qc.py                 quality checks: which outputs to trust
  pipeline.py           all stages on one document
  batch.py              every scan, one process each; crashes, runtime, memory, quality
  *_finder.py           bonus: find the regions without the supplied boxes
tools/                development: hand labelling, scoring against labels, evidence scripts
tests/                regression tests (the organisers' example and unit tests)
hack27/               organisers' material: slides, dataset, example (don't edit)
```

## Checking accuracy

Hand-labelled sections are in `tools/labels/` (see `tools/README.md`).

```
python -m tools.label_tool score tools/labels/<section>.txt   # boundaries, clusters, names
python -m tools.qc_eval                                       # evidence behind lithology/qc.py
```

## Workflow

- `git pull --rebase` before you push. Push small and often. Never force push.
- Keep to your own module where possible to avoid conflicts.
- Run `python -m pytest tests/` before pushing pipeline changes.
