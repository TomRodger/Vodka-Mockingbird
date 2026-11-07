"""Shared feature contracts for both layer patches and legend swatches."""
import warnings

import numpy as np
import pytest

from lithology.features import FeatureExtractor, pooled_fft


def patch(mask, colour=(70, 170, 230), ink=(0, 0, 0)):
    image = np.full((*mask.shape, 3), colour, dtype=np.uint8)
    image[mask] = ink
    return image


@pytest.mark.parametrize("colour", [(0, 0, 0), (255, 255, 255), (30, 60, 90)])
def test_solid_fill_has_colour_but_no_pattern(colour):
    image = patch(np.zeros((64, 64), dtype=bool), colour)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        f = FeatureExtractor().extract(image)
    np.testing.assert_array_equal(f["median_colour"], colour)
    assert not f["has_pattern"]
    assert not f["pattern_mask"].any()
    assert f["pattern_density"] == 0
    np.testing.assert_array_equal(f["fft_features"], np.zeros(24))


def test_scanner_noise_is_not_a_pattern():
    rng = np.random.default_rng(0)
    image = (150 + rng.integers(-3, 4, (128, 128, 3))).astype(np.uint8)
    f = FeatureExtractor().extract(image)
    assert not f["has_pattern"]
    assert f["pattern_density"] == 0
    assert np.max(np.abs(f["median_colour"].astype(int) - 150)) <= 1


@pytest.mark.parametrize("colour", [(70, 170, 230), (30, 60, 90)])
def test_dense_dark_pattern_keeps_true_background(colour):
    y, _ = np.indices((128, 128))
    mask = y % 8 < 5
    f = FeatureExtractor().extract(patch(mask, colour))
    np.testing.assert_array_equal(f["pattern_mask"] > 0, mask)
    np.testing.assert_array_equal(f["median_colour"], colour)
    assert f["pattern_density"] == pytest.approx(5 / 8)


def test_faint_real_lines_survive_noise_guard():
    y, _ = np.indices((128, 128))
    mask = y % 16 < 3
    image = patch(mask, (220, 220, 220), (200, 200, 200))
    f = FeatureExtractor().extract(image)
    np.testing.assert_array_equal(f["pattern_mask"] > 0, mask)
    assert np.linalg.norm(f["fft_features"]) > 0
    # A caller can deliberately preserve even fainter lines.
    image = patch(mask, (220, 220, 220), (214, 214, 214))
    assert not FeatureExtractor().extract(image)["has_pattern"]
    assert FeatureExtractor(min_contrast=2).extract(image)["has_pattern"]


def test_texture_distinguishes_direction_spacing_and_dots():
    y, x = np.indices((128, 128))
    masks = [y % 16 < 3, x % 16 < 3, y % 8 < 2,
             (y % 16 - 8) ** 2 + (x % 16 - 8) ** 2 <= 4]
    fe = FeatureExtractor()
    features = [fe.extract(patch(mask)) for mask in masks]
    for f in features:
        assert f["fft_features"].shape == (24,)
        assert np.isfinite(f["fft_features"]).all()
        assert f["fft_features"].sum() == pytest.approx(1)
        assert fe.build_vector(f).shape == (28,)
    for f in features[1:]:
        assert np.linalg.norm(features[0]["fft_features"] - f["fft_features"]) > .1
    base = features[0]["fft_features"]
    np.testing.assert_allclose(base, fe.extract(patch(masks[0][:64]))["fft_features"], atol=1e-7)
    shifted = fe.extract(patch(np.roll(masks[0], 5, axis=0)))["fft_features"]
    assert np.linalg.norm(base - shifted) < np.linalg.norm(base - features[1]["fft_features"])


def test_fft_bin_configuration_is_not_reused_from_another_call():
    y, _ = np.indices((64, 64))
    mask = y % 16 < 3
    expected = pooled_fft(mask, nr=3, na=3)
    pooled_fft(mask)
    actual = pooled_fft(mask, nr=3, na=3)
    assert actual.shape == (9,)
    np.testing.assert_allclose(actual, expected)


def test_old_saved_vector_has_clear_migration_error():
    f = FeatureExtractor().extract(patch(np.zeros((32, 32), dtype=bool)))
    f["fft_features"] = np.zeros(256)
    with pytest.raises(ValueError, match="recompute old saved features"):
        FeatureExtractor().build_vector(f)


@pytest.mark.parametrize("image", [np.zeros((0, 3, 3), np.uint8),
                                   np.zeros((8, 8), np.uint8),
                                   np.zeros((8, 8, 3), float)])
def test_invalid_image_has_clear_error(image):
    with pytest.raises(ValueError):
        FeatureExtractor().extract(image)
