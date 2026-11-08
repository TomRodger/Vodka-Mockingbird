# lithology: the pipeline

One module per stage. All stages pass intervals around as **pixel rows of the ORIGINAL full
image** (`top_y`, `bottom_y`); nothing resizes coordinates. Run everything from the repo root.

| module | stage | run |
|---|---|---|
| `segment.py` | 1. find interval boundaries in the lithology column | `python -m lithology.segment IMAGE JSON [PREFIX]` |
| `cluster.py` | 2. group intervals by visual pattern | `python -m lithology.cluster IMAGE JSON [--intervals CSV]` |
| `legend_extractor.py` | 3a. cut the legend into named swatches | `python -m lithology.legend_extractor LEGEND.png OUT_DIR` |
| `legend_match.py` | 3b. name each cluster from its document's legend | `python -m lithology.legend_match IMAGE JSON MERGED_CSV` |
| `depth/` | 4. read the depth ruler, convert rows to depths | see `depth/README.md` |
| `depth/depth_finder.py` | bonus: find the depth ruler without its annotation box | `python -m lithology.depth.depth_finder IMAGE` |
| `pipeline.py` | all stages on one document: the brief's CSV + annotated image | `python -m lithology.pipeline IMAGE JSON` |
| `batch.py` | every dataset scan, one process each; logs crashes, time, memory | `python -m lithology.batch` |
| `features.py` | shared features: column intervals *and* legend swatches | |
| `data.py` | dataset paths, annotation loading, cropping | |
| `output.py` | CSV and image writers | |
| `evaluate.py` | scoring against truth; the example check | `python -m lithology.evaluate` |

`pipeline.py` writes `<scan>.csv` (`top_y, bottom_y, cluster_id, lithology, top_depth,
bottom_depth, base_depth`) and `<scan>_annotated.png`, drawn on the original scan (tall logs
split into `_p01`, `_p02`, ... pages at full resolution). Legend matching needs Tesseract
(`brew install tesseract`, `pip install pytesseract`); without it every rock is `unknown`.

Use `--auto-depth` on the pipeline or depth extractor to locate the depth column
from the scan. Multiple plausible rulers require an explicit `--depth-candidate`
choice; check MD/TVD headings. Other pipeline boxes still come from the JSON.
See `depth/README.md` for standalone commands and limits.

# Legend matching (stage 3b)

Names each cluster from the document's own legend, so a new legend layout needs no new code.
For each cluster it takes the cluster's tallest interval (most reliable texture) and compares
it with every swatch the extractor found:

1. **Colour leads.** Column and legend are on the same scan, drawn by the same software, so a
   rock's fill colour matches its swatch closely. No swatch within ΔE 25 gives `unknown`
   instead of a forced wrong name.
2. **Mark shapes break colour ties.** Many legends have several rocks with the same fill
   (ardennes: Shale, Clay-shale, Marl, Sandy Clay-Shale are all green; black-and-white logs:
   every swatch is white). Among swatches within ΔE 8 of the best, the closest *mix of mark
   shapes* wins (see below).
3. **The commoner rock wins exact ties.** In the example legend Shale and Clay are both plain
   horizontal dashes, so the drawing cannot separate them. A fixed "most common in mudlogs"
   order decides. This is a stated prior, not a measurement.

Blank paper (`cluster_id -1`) is never matched: it would get the palest swatch's name.
Drilling symbols in legends (casing, cores, DST, mud loss/gain, no sample) are filtered out,
and OCR names are cleaned (junk symbols stripped, near-certain misreads like "Slity" fixed;
legend-specific names kept exactly as written, since the brief asks for the legend's name).

## Why mark shapes and not the pooled FFT

Clustering still uses the pooled FFT. Legend matching does not, because the two compare
different things:

- **Clustering compares the column with itself.** Every interval is drawn by the same
  software at the same scale and spacing, so shale at row 5,000 looks exactly like shale at
  row 9,000. A spacing-based descriptor is the right tool (example: row ARI 0.988).
- **Legend matching compares the column with the legend, and they are drawn differently.**
  On the example, column shale is rows of long dashes and legend shale is staggered short
  dashes; column sandstone is dots in neat rows, legend sandstone is a dense dot grid; and
  the scale differs. The FFT measures exactly what changes between the two drawings.

Measured on the example (17 true intervals, 3 rocks, legend of 10 swatches):

| method | right rock |
|---|---|
| pooled FFT | 0/17 (the correct swatch ranked 2nd to 5th at every scale tried) |
| pooled FFT + best legend scale (0.25x to 3.6x, one scale per document) | 0/17 |
| template matching (normalised cross-correlation, 0.8x to 1.25x) | wrong; best overlaps only 0.15 to 0.43 |
| **mark-shape mix** | **17/17** |

What stays the same between the two drawings is the *kind* of mark, not its spacing.
`mark_shapes()` labels each connected ink mark as dot (solid, roughly round, any size),
blob (L, T, triangle, cross symbols: mostly empty inside their box), horizontal, vertical,
long line or grid, drops border and frame lines, and describes a patch by the fraction of
marks in each class. The fractions don't depend on scale or spacing.

Rule of thumb: **FFT when comparing like with like, shapes when comparing two different
drawings of the same rock.**

Idea for later (unmeasured): the shape mix ignores where marks sit horizontally, which is
what splits one rock into several clusters on colour logs (ardennes: one green shale in 5
clusters). It may help clustering too. Test it with row ARI on labelled sections first.

## Results across the dataset (final run)

51 of 51 real scans, 0 crashes (the 52nd file is a byte-identical `(1)` duplicate with no
annotation). 11,999 intervals, 6.5% `unknown`, 30 documents with none. 8 missing depth
values. Median 25 s and 0.6 GB per document (max 131 s, 4.1 GB). Accuracy beyond the example
is checked by eye only until labelled sections exist: `python -m tools.label_tool score`.

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

`--texture pooled` (default) uses the shared extractor's 24-value pooled FFT.
`--texture team` remains accepted for compatibility and now uses the same descriptor
and thresholds; it no longer selects the old 16×16 spectrum.

## Shared features for layers and legend swatches

`FeatureExtractor.extract(image)` takes a non-empty **uint8 BGR** image and returns:

| field | meaning |
|---|---|
| `median_colour` | 3 background colour values in BGR order |
| `pattern_mask` | 2-D uint8 mask: 255 for dark linework, 0 for background |
| `pattern_density` | fraction of pixels marked as linework |
| `fft_features` | **24** pooled frequency values describing spacing and direction |
| `has_pattern` | whether the patch passed the extractor's contrast check |

`build_vector(features)` returns **28 values**, replacing the old 260-value vector:
3 normalised BGR colours, 1 density and 24 texture values. Rebuild saved feature
vectors if they used the old representation. Layers and legend swatches must use
the same descriptor; these features do not themselves identify rock names.

Solid fills, including pure black, keep their background colour and get zero mask,
density and texture. Small grey-level variations are treated as scanner noise.
The default minimum contrast is 12 levels out of 255; use
`FeatureExtractor(min_contrast=...)` deliberately when processing very faint patterns.
Clustering additionally keeps its own stricter no-pattern/blank/thin-interval guards.

## Current results

With the band-based segmentation (`segment.py`, see its comments), on the example (the only labelled data):

| input | texture | clusters | row ARI | boundary recall / precision (±20 px) |
|---|---|---|---|---|
| true intervals | pooled | 3 | **1.000** | 1.00 / 1.00 |
| segmentation | pooled | 3 | **0.988** | 1.00 / 1.00 |

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

- **Very faint patterns:** the extractor's contrast guard can suppress real lines
  below its configured threshold. Lower it only after checking plain/noisy fills too.
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
