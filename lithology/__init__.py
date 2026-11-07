"""Lithology column extraction pipeline (Viridien challenge).

Stages, each its own module, all working in pixel rows of the ORIGINAL full image:
    segment   find interval boundaries in the interpreted lithology column
    cluster   group intervals by visual pattern (uses features)
    depth     read the depth ruler and convert pixel rows to depths (lithology/depth/)
Support:
    data      dataset paths, annotation loading, cropping
    features  shared feature extraction (column intervals and legend swatches)
    output    CSV and image writers
    evaluate  scoring against truth and the example check

Run any stage as a module from the repo root, e.g. `python -m lithology.evaluate`.
"""
