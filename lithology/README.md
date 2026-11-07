# lithology: the pipeline

One module per stage. All stages pass intervals around as **pixel rows of the ORIGINAL full
image** (`top_y`, `bottom_y`); nothing resizes coordinates. Run everything from the repo root.

| module | stage | run |
|---|---|---|
| `segment.py` | 1. find interval boundaries in the lithology column | `python -m lithology.segment IMAGE JSON [PREFIX]` |
| `cluster.py` | 2. group intervals by visual pattern | `python -m lithology.cluster IMAGE JSON [--intervals CSV]` |
| `depth/` | 4. read the depth ruler, convert rows to depths | see `depth/README.md` |
| `features.py` | shared features: column intervals *and* legend swatches | |
| `data.py` | dataset paths, annotation loading, cropping | |
| `output.py` | CSV and image writers | |
| `evaluate.py` | scoring against truth; the example check | `python -m lithology.evaluate` |

Stage 3 (legend matching: a rock name per cluster) and the final CSV + annotated image the
brief asks for are not built yet. Legend matching should use `features.py` and the
`*_representatives.json` that clustering writes.

# Clustering (stage 2)

Groups lithology intervals by visual pattern. Takes intervals from `lithology.segment`
(or any CSV with `top_y,bottom_y`), uses the shared `FeatureExtractor` in `features.py`,
and outputs a `cluster_id` and confidence per interval, merged intervals, and one
representative per cluster for legend matching.

```
python -m lithology.evaluate                                      # the example check
python -m lithology.cluster IMAGE JSON                            # segmentation -> clustering
python -m lithology.cluster IMAGE JSON --intervals my.csv --truth labels.csv
python -m lithology.cluster IMAGE JSON --y0 9053 --y1 11053       # a section only
```

## Outputs (in `out/`, git-ignored)

| file | contents |
|---|---|
| `*_intervals.csv` | every INPUT interval: `top_y, bottom_y, cluster_id, margin, flags` |
| `*_merged.csv` | after joining touching same-cluster intervals, plus `n_pieces` |
| `*_representatives.json` | per cluster: most typical member, member count, Lab colour. **Handoff to legend matching.** |
| `*_overlay.png` | the column with a cluster-coloured bar and boundaries |
| `*_montage.png` | one row per cluster, least typical members first; red border = thin interval |

`margin`: distance to the cluster's most typical member. 0 is typical; above 0.7 it's
flagged `low_margin`. Other flags: `thin` (under 24 px, attached by colour/neighbours),
`no_pattern` (solid fill), `colour_ambiguous`, `new_from_thin`, and `no_data`.

`cluster_id = -1` with flag `no_data` means **blank paper**: no rock was logged there (white,
no pattern). It is left out of clustering and has no representative, so legend matching never
gives it a rock name. The example's column is blank below row 10432.

## How it works

1. **Describe** each interval: Lab fill colour, plus texture from the extractor's pattern
   mask. Solid fills get an empty texture of their own, rather than Otsu's noise. Blank
   intervals (judged on the patch's inner part, away from the column's thick borders) are
   set aside as `no_data`.
2. **Distance** = `max(colour ΔE / 10, texture / document noise unit)`. Close only if
   both agree. Texture noise is handled in two ways:
   - **per pair**: each distance is divided by `√(1/tiles_i + 1/tiles_j)`, so short
     intervals (few 32-row tiles, noisy texture) get more tolerance than tall ones
   - **per document**: the unit is 3 × the median distance to each interval's 3rd-nearest
     neighbour, because noise differs ~6× between the example and Rio Nueces. (1st-nearest
     was fooled by near-duplicate intervals; 4th-nearest crossed into other rocks on the example.)
3. **Average-linkage** clustering, cut at 1.0, on intervals ≥ 24 px.
4. **Thin intervals** attach by colour, or by their nearest neighbour's cluster when several
   clusters share the colour (always the case in black-and-white logs).
5. **Merge** touching same-cluster intervals, re-describe the merged ones and repeat
   (up to 3 rounds).

`--texture pooled` (default) is a windowed, tiled FFT pooled into 24 bins, computed on the
extractor's mask. `--texture team` uses the extractor's own 16×16 `fft_features`.

## Current results

With the band-based segmentation (`segment.py`, see its comments), on the example (the only labelled data):

| input | texture | clusters | row ARI | boundary recall / precision (±20 px) |
|---|---|---|---|---|
| true intervals | pooled | 3 | **1.000** | 1.00 / 1.00 |
| true intervals | team | 1 | 0.000 | – |
| segmentation | pooled | 3 | **0.988** | 1.00 / 1.00 |
| segmentation | team | 1 | 0.000 | – |

Before the segmentation rewrite, segmentation + pooled scored row ARI 0.100 (it found 5/16 boundaries).

Segmentation on the whole example column gives 19 intervals: 16 cuts within 0–6 px of a true boundary,
the end of logged rock found 17 px late (row 10432 vs 10415; blank paper below), and one false
cut at row 6356, where a siltstone pattern visibly fades. Clustering merges that one away.

Row ARI gives every pixel row its cluster and compares with the true lithology of that row,
so it scores the result independently of how the column was cut. Boundaries use ±20 px
because `example.csv` snaps them to the 10 ft depth grid, up to ~13 px from the visible edge.

Unlabelled colour sections (checked by eye in the montages):

| section | intervals | clusters | what the montage shows |
|---|---|---|---|
| Rio Nueces 212, rows 9053–11053 | 50 | 10 | two clean main clusters (shale, limestone); the rest are mostly thin slivers |
| mc761 1md, rows 10335–11135 | 31 | 3 | not inspected yet |
| Shell 08852, rows 1769–3769 | 13 | 2 | not inspected yet |
| ocs-g 26419, rows 61177–63177 | 19 | 3 | not inspected yet |
| Critter Creek 201, rows 39730–40930 | 29 | 3 | not inspected yet |

## Known limits

- **The team 16×16 FFT doesn't separate the example's rocks.** Same-rock and
  different-rock distances overlap, so everything lands in one cluster. Use `pooled`.
- **Thin slivers make small extra clusters.** Pieces under 24 px that contain a black
  boundary line or brick row get a skewed colour, so they don't attach to their rock (Rio's
  8 small clusters). Likely fix: drop near-black rows before measuring a thin piece's fill.
- **Blank detection is simple**: white (Lab L* ≥ 95, |a|,|b| < 5) and no pattern. A real rock
  drawn as plain white with no pattern would be marked `no_data`. None was in the sections tried.
- **Thresholds are untuned.** ΔE 10, cut 1.0, 24 px, noise ×3 / 3rd neighbour, contrast 40
  are defaults chosen on the example plus visual checks. Tune on hand-labelled sections
  (row ARI).
- **Memory is Pillow's, not ours.** Pillow decodes the whole compressed page when cropping:
  ~3.6 GB peak on the tallest log (1680 × 313,800). Strip-wise TIFF reading (e.g. `tifffile`)
  would fix it for every script.
- **Strip-based clustering** (cluster fixed-height strips, then vote) was prototyped before
  the segmentation rewrite. It's no longer needed for the example, but might help documents where
  segmentation still misses boundaries.
