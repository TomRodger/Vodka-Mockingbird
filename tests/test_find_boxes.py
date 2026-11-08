"""lithology.find_boxes: joining the bonus finders (the finders themselves have their own tests)."""

import pytest

import lithology.find_boxes as F
from lithology.data import load_annotation


def test_legend_box_converted_to_supplied_format():
    assert F.xywh_to_xyxy((10, 20, 30, 40)) == [10, 20, 40, 60]


def test_iou():
    assert F.iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    assert F.iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0
    assert F.iou([0, 0, 10, 10], [5, 0, 15, 10]) == pytest.approx(50 / 150)


def test_found_boxes_load_like_supplied_ones(tmp_path):
    boxes = {"interpreted_lithology_column": [1, 2, 3, 4], "lithology_legend": [5, 6, 7, 8],
             "depth_column": None}
    path = tmp_path / "found.json"
    F.write_annotation(path, "scan.tif", boxes)
    assert load_annotation(path) == {"interpreted_lithology_column": [1, 2, 3, 4], "lithology_legend": [5, 6, 7, 8]}
    import json
    assert {"width", "height", "bounding_boxes"} <= set(json.load(open(path)))   # depth reader needs these


def test_failed_finder_costs_its_region_only_and_is_reported(tmp_path, monkeypatch):
    def broken(image):
        raise ValueError("no ruler")
    monkeypatch.setattr(F, "FINDERS", {"interpreted_lithology_column": lambda i: ([1, 2, 3, 4], "ok"),
                                       "lithology_legend": lambda i: ([5, 6, 7, 8], "ok"),
                                       "depth_column": broken})
    boxes, notes, _ = F.find_boxes("scan.tif")
    assert boxes["depth_column"] is None and "no ruler" in notes["depth_column"]
    assert boxes["lithology_legend"] == [5, 6, 7, 8]
    with pytest.raises(RuntimeError, match="depth_column"):
        F.annotation_for("scan.tif", tmp_path / "found.json")
