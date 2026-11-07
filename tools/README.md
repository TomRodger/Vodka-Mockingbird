# tools: labelling and descriptor testing (development only, not part of the pipeline)

Decides between FFT and HOG (plus variants) for describing lithology patterns,
using hand-labelled sections of real columns. Everything runs on a laptop CPU.

| file | what it is |
|---|---|
| `texture_features.py` | FFT and HOG variants compared by `texture_test.py` (the pipeline's own features are in `lithology/features.py`) |
| `label_tool.py` | renders a column section with a pixel ruler, drafts boundaries, validates your labels |
| `texture_test.py` | runs the comparison over all labelled sections and prints a verdict |
| `labels/` | hand-made label files: **ground truth, commit them**. Ships with the organisers' example. |

## 0. Setup (once)

```
cd <repo root>                     # ALWAYS run commands from here
pip install -r requirements.txt
```

## 1. Sanity check on the example (1 minute)

```
python -m tools.texture_test
```

Expected: `ari_int` is 1.000 for every descriptor, followed by the NOTE line saying
the section is too easy to separate them. That's correct, and it's why you need
step 2.

## 2. Label six sections

These six are from six different wells (siblings of the same well would inflate the
scores), chosen to cover what the example lacks:

| # | why this one | command |
|---|---|---|
| 1 | black-and-white, 4 textures incl. two dot patterns (only B&W well, sibling of the example) | `python -m tools.label_tool show 0000000009_d-582_0000020435-0001_log02 --y0 4387 --height 2000` |
| 2 | many thin beds, 4–5 colours, column 79 px | `python -m tools.label_tool show 60817411400100_mc761_1_st1_mudlog_1md --y0 10335 --height 800` |
| 3 | shale vs limestone interbeds, column 70 px | `python -m tools.label_tool show Rio_Nueces_3744_Unit_1_212 --y0 9053 --height 2000` |
| 4 | dashed shale vs cyan vs sand, column 65 px | `python -m tools.label_tool show 60812400480000 --y0 61177 --height 2000` |
| 5 | thin sands in shale, column 60 px | `python -m tools.label_tool show Shell_08852 --y0 1769 --height 2000` |
| 6 | onshore vendor: brown shale / white patterned / orange | `python -m tools.label_tool show "Critter Creek 201" --y0 39730 --height 1200` |

Each `show` writes:
- `out/views/<doc>__y<row>_pNN.png`: the section, zoomed. Left to right:
  pixel-row ruler, depth column, lithology column, draft boundaries in red with
  their row numbers.
- `out/views/<doc>__legend.png`: the legend, for exact names.
- `tools/labels/<doc>__y<row>.txt`: your label file, pre-filled with the
  draft boundaries and `?` for every lithology.

`show` never overwrites a label file you've started editing (`--force` does).

### How to fill in the label file

Open the `.txt` next to the view pages. Each line starts an interval:

```
10335 Shale
10364 Siltstone
10374 Shale
10428 Sandstone
11135 END
```

Rules:
1. **Row** = the first pixel row of the new rock, read off the ruler (ticks every
   10 rows, numbers every 50). Within ~3 px is fine; the test trims 4 px off each
   edge anyway.
2. **Name** = exactly as written in the legend (`Shale`, not `shale` or `Sh`). The
   test treats different spellings as different rocks.
3. **Fix the draft**: delete a line to remove a false boundary, add a line for a
   missed one. The draft only sees colour changes, so it misses texture-only
   changes and finds nothing in black-and-white documents (section 1 is all manual).
4. **Unsure what rock it is?** Write `UNSURE`. The test skips those intervals.
   Don't guess: a wrong label is worse than a missing one.
5. **Label thin beds too.** The 64 px test skips them, but the 32 and 16 px tests use
   them, and you'll want them for evaluating the real pipeline later.
6. The last line is `<row> END`, the bottom of the section.

Expect 10–20 minutes per section. Section 2 is dense: do it last.

## 3. Check each label file

```
python -m tools.label_tool check "tools/labels/<doc>__y<row>.txt"
```

This validates the file (rows increasing, no `?` left, inside the column), writes
the matching `.csv` the test reads, and renders `out/views/<doc>__y<row>_check_pNN.png`
with your labels as coloured bars. **Open those pages and look**: a misplaced line
or a wrong name is obvious at a glance. It also prints how many intervals are tall
enough for each window size.

Use quotes around paths: several document names contain spaces.

## 4. Run the comparison

```
python -m tools.texture_test
```

It uses every `tools/labels/*.csv`, at window heights 64, 32 and 16, with
seeds 0, 1, 2. It takes a few minutes. Per-run numbers go to `out/texture_results.csv`. For each setting it prints:

```
--- window height 32, texture + colour ---
     descriptor       ari_int    silhouette  nn_acc ari_win  shift  ms/win  best on
            fft   0.912±0.004   ...
VERDICT: ...
```

- **ari_int** (main metric): clustering intervals vs your labels. 1 = perfect, 0 = random.
- **±** is the spread across seeds: differences smaller than about twice this are noise.
- **silhouette**: separation margin; decides ties on ari_int.
- **best on**: how many sections this descriptor had the top ari_int on.
- **shift** (lower is better): sensitivity to where the pattern starts.

### The decision rule (fixed now, before seeing results)

1. Keep every descriptor whose `ari_int` is within 2× noise of the best. If one is
   left, it wins.
2. Among those, keep every one whose `silhouette` is within 2× noise of the best.
   If one is left, it wins.
3. Otherwise the fastest of those wins.

The script applies this automatically and prints the reasoning on the VERDICT line.

### How to read the result

- **"texture only" decides which descriptor is better.** Use the window height you
  will actually use. `check` prints how many intervals are tall enough for each size;
  in thin-bedded documents (mc761, Critter Creek) most are only usable at 16.
- **"texture + colour" decides whether it matters.** If every descriptor scores
  about the same once colour is added, colour is doing the work, and you should
  pick the fastest texture descriptor and say so.
- The verdict is only as good as the sections. If one descriptor wins on 5/6
  sections, that's a real result. If they split 3/3, report it as a tie.

## Known findings so far (worth a demo slide)

- **Columns are narrow.** Interpreted columns are 29–95 px wide, many under 50. Windows
  must be column-width, not square; the test handles this.
- **Not all patterns are black.** FREEMAN draws grey lines (~0.65) on a pale fill. A
  fixed black threshold sees no texture at all, so `binarise` is relative to the fill.
- **Some documents are colour-only.** FREEMAN's patterns are so faint that texture is
  empty after trimming the border; only colour separates its rocks.
- **Many beds are thin.** In mc761 most intervals are 10–30 px tall, and in a Critter
  Creek section only 7 of 39 intervals fit a 32 px window. Texture must work on
  ~16 px windows, or colour carries those intervals alone.
- **Standard HOG is position-dependent.** On the example, moving a window a few px
  changes it almost as much as changing rock (shift 0.86). Pooled HOG and FFT don't
  have this problem.
- **Not covered yet:** the "same texture, different colour" case (e.g. light vs dark
  grey shale in `60817411400200_mc761..._bp01`, rows ~13263–14000). It has too few
  intervals in one section to score; if you find a busier section with shale
  variants, label it and add it.
