"""Check automatic location without requiring OCR models or dataset files."""

import io
import json
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from lithology.depth.depth_finder import (
    DepthColumnError, _read_tile, choose_column, find_depth_column, rank_columns,
)


def ruler(x=100, depth0=1000):
    return [dict(pixel_x=x, pixel_y=500 + i * 400, depth=depth0 + i * 20,
                 left=x - 20, right=x + 20, label_height=20,
                 confidence=0.99, text=str(depth0 + i * 20), ink="black",
                 band=i // 2, rotation=0) for i in range(8)]


def test_ruler_beats_repeated_graph_values_and_overlap_duplicates():
    rows = ruler()
    graph = [dict(r, pixel_x=500, left=480, right=520, depth=42) for r in rows]
    duplicate = [dict(r, pixel_y=r["pixel_y"] + 1) for r in rows]
    candidates = rank_columns(rows + graph + duplicate, (800, 4000))
    assert len(candidates) == 1
    assert candidates[0]["n_markers"] == 8
    assert candidates[0]["bbox"][0] < 80 < 120 < candidates[0]["bbox"][2]


def test_two_rulers_require_an_explicit_choice():
    candidates = rank_columns(ruler() + ruler(400, 900), (800, 4000))
    with pytest.raises(DepthColumnError, match="Several") as error:
        choose_column(candidates)
    assert len(error.value.candidates) == 2
    assert choose_column(candidates, 1) == candidates[1]
    with pytest.raises(DepthColumnError):
        choose_column(candidates, -1)


def test_no_ruler_and_short_header_sequences_are_rejected():
    assert not rank_columns([dict(r, pixel_y=100 + i * 15, band=0)
                             for i, r in enumerate(ruler())], (800, 4000))
    with pytest.raises(DepthColumnError, match="No reliable"):
        choose_column([])


@pytest.mark.parametrize("rotation", [0, 90, 270])
def test_tile_coordinates_survive_rotation_padding_and_crop(rotation):
    # A label in padded coordinates x=50..90,y=70..90 should land back
    # at full-image centre (150, 360), whatever orientation OCR sees.
    points = np.array([[50, 70], [90, 70], [90, 90], [50, 90]], float)
    if rotation == 90:
        points = np.column_stack((points[:, 1], 240 - points[:, 0]))
    elif rotation == 270:
        points = np.column_stack((340 - points[:, 1], points[:, 0]))

    def engine(image, **kwargs):
        return SimpleNamespace(txts=["1,200"], scores=[0.99], boxes=[points])

    rows = _read_tile(Image.new("RGB", (800, 900), "white"),
                      (100, 300, 300, 600), engine, rotation, 2)
    assert rows[0]["pixel_x"] == pytest.approx(150)
    assert rows[0]["pixel_y"] == pytest.approx(360)
    assert rows[0]["depth"] == 1200


def test_search_and_vertical_refinement_do_not_use_annotations(monkeypatch):
    from lithology.depth import depth_finder, extract_depths

    page = Image.new("RGB", (800, 4000), "white")
    monkeypatch.setattr(Image, "open", lambda _: page)

    def tile(image, box, engine, rotation, band):
        return [dict(r, band=band) for r in ruler()
                if box[1] <= r["pixel_y"] < box[3]]

    monkeypatch.setattr(depth_finder, "_read_tile", tile)
    monkeypatch.setattr(extract_depths, "read_ruler", lambda *a: ruler())
    result = find_depth_column("unused.tif", engine=object())
    assert result["bbox"][1:] == [100, 132, 3700]
    assert result["n_markers"] == 8
    assert result["depth_kind"] == "unconfirmed"


@pytest.mark.parametrize("annotated", [False, True])
def test_reader_can_extract_without_a_depth_box_or_override_it(monkeypatch, annotated):
    from lithology.depth import depth_finder, extract_depths

    saved = {}

    class MemoryPath:
        def __init__(self, name="memory"):
            self.name = name

        def __truediv__(self, name):
            return MemoryPath(self.name + "/" + name)

        def mkdir(self, **kwargs):
            pass

        def write_text(self, text, **kwargs):
            saved[self.name] = text

        def open(self, *args, **kwargs):
            return io.StringIO()

    page = Image.new("RGB", (800, 4000), "white")
    monkeypatch.setattr(Image, "open", lambda _: page)
    monkeypatch.setattr(Image.Image, "save", lambda *args, **kwargs: None)

    def find(image, **kwargs):
        assert kwargs["candidate_index"] == 1
        return {"bbox": [68, 100, 132, 3700]}

    monkeypatch.setattr(depth_finder, "find_depth_column", find)

    def read(image, box, *args):
        assert box == (68, 100, 132, 3700)
        return ruler()

    monkeypatch.setattr(extract_depths, "read_ruler", read)
    # A deliberately wrong annotation proves --auto-depth does not peek at it.
    annotation = SimpleNamespace(read_text=lambda **kw: json.dumps({
        "width": 800, "height": 4000,
        "bounding_boxes": {"depth_column": [500, 10, 700, 200]}})) if annotated else None
    markers = extract_depths.extract("unused.tif", annotation, MemoryPath(),
                                     engine=object(), depth_candidate=1, auto_depth=annotated)
    assert len(markers) == 8
    assert json.loads(saved["memory/depth_location.json"])["bbox"] == [68, 100, 132, 3700]


def test_pipeline_forwards_automatic_choice_and_adds_depths(monkeypatch):
    from lithology import pipeline

    def extract(image, annotation, directory, **kwargs):
        assert kwargs == {"auto_depth": True, "depth_candidate": 1}
        return ruler()

    monkeypatch.setattr(pipeline, "extract_markers", extract)
    monkeypatch.setattr(pipeline.shutil, "rmtree", lambda *args, **kwargs: None)
    rows, error = pipeline.add_depths([{"top_y": 500, "bottom_y": 900}],
                                     "unused.tif", "unused.json", "unused", True, 1)
    assert error is None
    assert rows[0]["top_depth"] == 1000
    assert rows[0]["base_depth"] == rows[0]["bottom_depth"] == 1020


def test_pipeline_memory_reporting_works_on_this_platform():
    from lithology.pipeline import peak_memory_mb

    assert peak_memory_mb() > 0
