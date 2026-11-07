# Vodka Mockingbird

Viridien challenge (HackTheBurgh): automatically extract lithology intervals, their rock
types and depths from scanned well logs.

## Layout

```
lithology/        the pipeline: one module per stage (see lithology/README.md)
  segment.py        1. interval boundaries in the lithology column
  cluster.py        2. group intervals by visual pattern
  depth/            4. read the depth ruler, convert pixel rows to depths
  features.py       shared features (column intervals and legend swatches)
  data.py           dataset paths, annotation loading, cropping
  output.py         CSV and image writers
  evaluate.py       scoring against truth; the example check
tools/            development tools: labelling ground truth, comparing texture descriptors
tests/            regression check on the organisers' example
hack27/           organisers' material: slides, dataset, example, notebook (don't edit)
backend/ frontend/  web app scaffold
out/              generated outputs (git-ignored)
```

Not built yet: stage 3 (legend matching: a rock name per cluster) and the final CSV +
annotated image per document.

## Running the pipeline

From the repo root:

```
pip install -r requirements.txt
python -m lithology.evaluate                         # example check, ~3 s
python -m lithology.segment IMAGE ANNOTATION_JSON    # stage 1 on one scan
python -m lithology.cluster IMAGE ANNOTATION_JSON    # stages 1-2 on one scan
pytest tests/                                        # does the example still score?
```

Modules are run with `python -m lithology.<module>`, not `python lithology/<module>.py`:
they import each other as a package.

## Running the web app

Needs [uv](https://docs.astral.sh/uv/) and Node 20+.

**Backend** (http://localhost:8000, docs at /docs):

```
cd backend
cp ../.env.example .env
uv sync
uv run uvicorn app.main:app --reload
```

**Frontend** (http://localhost:5173), in a second terminal:

```
cd frontend
npm install
npm run dev
```

The page should say `Backend: ok`. If it doesn't, the backend isn't running.

## Workflow

- `git pull --rebase` before you push. Push small and often.
- Don't force push.
- Keep to your own module where possible to avoid conflicts.
- Run `pytest tests/` before pushing pipeline changes.
- Format before committing: `uv run ruff format` in `backend/`.
