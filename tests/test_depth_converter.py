"""DepthConverter: default extrapolation allowance and NaN fallback."""

import math

import pytest

from lithology.depth.depth_converter import DepthConverter

# Labels every 400 px / 20 ft, like the example's ruler.
MARKERS = [(1000, 100.0), (1400, 120.0), (1800, 140.0), (2200, 160.0)]


def test_default_allows_one_missing_label():
    c = DepthConverter(MARKERS)                    # 1.5 gaps = 600 px
    assert c.max_extrapolation_pixels == 600
    assert c.depth_at(600) == pytest.approx(80.0)  # one label above the first marker


def test_explicit_zero_forbids_extrapolation():
    with pytest.raises(ValueError):
        DepthConverter(MARKERS, max_extrapolation_pixels=0).depth_at(999)


def test_out_of_range_boundary_is_nan_not_crash():
    r = DepthConverter(MARKERS).convert_interval({"top_y": 100, "bottom_y": 1200})
    assert math.isnan(r["top_depth"])              # 900 px above: too far
    assert r["base_depth"] == pytest.approx(110.0)
    assert r["depth_extrapolated"] is True


def test_example_first_interval():
    # Markers OCR'd from the organisers' example: first readable label is 25420 at
    # row 4804.5, so the first true interval (top_y 4409 = 25400 ft) needs extrapolation.
    c = DepthConverter([(4804.5, 25420.0), (5205.0, 25440.0), (5605.5, 25460.0)])
    r = c.convert_interval({"top_y": 4409, "bottom_y": 4809})
    assert r["top_depth"] == pytest.approx(25400, abs=0.5)
    assert r["base_depth"] == pytest.approx(25420, abs=0.5)
