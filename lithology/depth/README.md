# Depth: turn layer positions into depths

This is our part of the challenge: find the top and base depth of each rock
layer. Base means bottom.

## What I need from the team

For each document:

1. The original scan (.tif or .tiff).
2. Its supplied annotation JSON, or use the automatic depth-column finder below.
3. Your detected layers, with `top_y` and `bottom_y` for each layer.

Send layer positions as pixel rows in the ORIGINAL full image. If you crop or
resize the image, convert the positions back before sending them.
Include which scan the layers belong to. I do not need rock names or cluster
labels yet, but the code keeps those fields if you include them.

Your layer data can be a CSV:

```csv
top_y,bottom_y,cluster_id
4409,4809,A
4809,5010,B
```

These rows just illustrate the format. We can connect to a Python list of
rows instead once your code is ready; you do not have to save a CSV.

## What my code does

1. `extract_depths.py` reads the printed numbers on the scan's depth ruler.
   It saves which image row goes with each depth in `markers.csv`.
2. `depth_converter.py` uses that ruler to calculate the depth at your
   `top_y` and `bottom_y` positions.
3. It returns your rows with `top_depth` and `base_depth` added.

For example, a position halfway between two depth labels gets a depth halfway
between their values. Your layer boundaries and any cluster/rock labels are
kept. If you already have a `bottom_depth` field, it is updated too.

The code reads depths; your code supplies the layer boundaries.

If a column has a black main ruler and a coloured second ruler, the default is
the black main ruler. This keeps the two sets of numbers separate. Check the
heading to confirm whether that ruler is MD or TVD; colour alone does not tell
us that. `--series secondary` reads the coloured ruler instead when needed.
If the only ruler is coloured, the default keeps it.

The extractor also tries rotating the ruler if normal OCR fails. It converts
rotated positions back to the original image rows automatically. Repeated or
backwards OCR readings are removed only when enough readings still agree.
If needed, it tries shorter crops so nearby labels do not get joined together.
On rulers with repeated abbreviated numbers such as `20,40,60,80`, it uses the
full printed labels, such as `1500,1600`, as anchors instead. It does not guess
the missing thousands. Slight scan stretch is checked in local sections;
abrupt changes in scale are still rejected.

## Run from the terminal

From the repository root, install the pipeline's dependencies once (they include
this folder's OCR packages):

```powershell
py -m pip install -r requirements.txt
```

## Bonus: find the depth column automatically

From the repository root, this searches the scan and prints a box without
using the annotation or saving any files:

```powershell
py -m lithology.depth.depth_finder "path/to/scan.tif"
```

To find the column AND read its depth labels:

```powershell
py -m lithology.depth.extract_depths --image "path/to/scan.tif" --auto-depth --output-dir out/depth/my_scan
```

The finder looks for vertically aligned numbers which increase consistently.
It searches several strips across the whole page at their original resolution,
then reads the chosen narrow column to find its top and bottom. It uses the
same RapidOCR dependencies as our existing reader. It does not need Tesseract.

The output includes `depth_location.json`, containing the detected box,
alternatives and a reminder to check the ruler heading. Box coordinates are
`[left, top, right, bottom]` in the original full image.

If two rulers look equally plausible, it reports their candidate boxes instead
of guessing. Check the MD/TVD headings and rerun with `--depth-candidate 0`
(or `1`, etc.). Finding a ruler does not establish its units or MD/TVD meaning.
If no reliable ruler is found, use the supplied annotation as before.

The full pipeline also accepts `--auto-depth`. It still needs the annotation
for the lithology column and legend; only the depth box becomes automatic.
Automatic location is an optional heuristic, not a guaranteed replacement for
all dataset annotations. Tall TIFFs still have the existing Pillow memory cost.

Initial live checks found a usable ruler on four scans: the organisers' example,
`md5_mud_show_2`, `mc761 ... 1tvd`, and `gb506#2`. Detected boxes include padding
around labels and can differ from the hand-drawn boxes. This is not a location
accuracy score for the full dataset.
An end-to-end automatic run on the example read 16 ruler labels and converted
all 17 reference intervals: average absolute depth difference 0.12 ft, maximum
0.19 ft, using the converter's default bounded extrapolation at the top edge.

From Python:

```python
from pathlib import Path
from lithology.depth.depth_finder import find_depth_column
from lithology.depth.extract_depths import extract

location = find_depth_column(Path("scan.tif"))
print(location["bbox"])
markers = extract(Path("scan.tif"), None, Path("out/depth/my_scan"))
```

## Use the supplied annotation

Then, from this folder (`lithology/depth`), replace the placeholder paths with your files:

```powershell
cd lithology/depth
py extract_depths.py --image "path/to/scan.tif" --annotation "path/to/scan.json" --output-dir results/my_scan
py depth_converter.py --markers results/my_scan/markers.csv --intervals "path/to/layers.csv" --output results/my_scan/layers_with_depths.csv
```

Use a new output folder/name for each run. Paths are relative to the terminal's
current folder unless you use a full path.

The output folder contains the markers, an image preview for checking OCR,
the raw OCR readings for troubleshooting, and your layers with depths added.
OCR means reading text from an image. Dependencies and results are ignored by Git.
`diagnostics.json` records which orientation worked and which series was requested.

## Use from Python

From the repository root:

```python
from pathlib import Path
from lithology.depth.extract_depths import extract
from lithology.depth.depth_converter import DepthConverter

# Read the ruler once for this scan.
markers = extract(Path("scan.tif"), Path("scan.json"), Path("results/my_scan"))
converter = DepthConverter((m["pixel_y"], m["depth"]) for m in markers)

# layers is the list of rows from your layer-detection code.
layers_with_depths = [converter.convert_interval(layer) for layer in layers]
```

## Things to know

- Use the same scan for the ruler and the layer positions. Depths keep the
  ruler's units and scale (for example feet and MD or TVD).
- Blurry or cut-off labels can be missed. The converter normally allows an
  extension of 1.5 typical label gaps. Beyond that, a boundary depth is `NaN`.
- Choose an explicit limit with `--max-extrapolation-pixels N` on the converter,
  or `max_extrapolation_pixels=N` in Python. Set N to 0 to disable extensions.
  Layers outside readable labels get `depth_extrapolated=True`.
- Validation covered all 51 unique dataset scans: each passed calibration
  checks. Twelve focused checks also passed. This is not an accuracy score:
  inspect the selected scale and allow for OCR/tick-position errors.
- OCR uses a 1200-pixel detector size limit to avoid unnecessarily enlarging
  ruler crops. The reference example's maximum depth difference was 0.33 feet.
