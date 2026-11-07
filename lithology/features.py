"""Shared feature extraction for lithology patches (column intervals and legend swatches).

Input convention: every function takes a BGR uint8 image (cv2.imread order). If you
crop with Pillow (RGB), convert with `arr[:, :, ::-1]` first, otherwise red and blue
swap and colour matches between the column and the legend break silently.

    FeatureExtractor.extract(img) -> pattern mask, fill colour, density, 24 pooled FFT values
    bgr_to_lab(bgr)               -> real Lab colour, where distance ~ perceived difference
    pooled_fft(mask)              -> tiled, windowed FFT pooled into 24 spacing x direction bins
"""

import cv2
import numpy as np


class FeatureExtractor:
    """Describe dark linework on a fill colour, without naming the rock.

    min_contrast is the minimum useful grey-level spread (0-255). The default
    ignores small scanner noise; lower it deliberately for very faint patterns.
    """

    def __init__(self, min_contrast=12):
        if not np.isfinite(min_contrast) or not 0 <= min_contrast <= 255:
            raise ValueError("min_contrast must be a finite number from 0 to 255.")
        self.min_contrast = min_contrast

    def extract(self, img):
        """Return BGR background colour, mask, density and a 24-value texture."""
        if not isinstance(img, np.ndarray) or img.ndim != 3 or img.shape[2] != 3 or img.size == 0:
            raise ValueError("Expected a non-empty H x W x 3 BGR image.")
        if img.dtype != np.uint8:
            raise ValueError("Expected a uint8 BGR image with values from 0 to 255.")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        low, high = np.percentile(gray, [0.1, 99.9])
        has_pattern = high > low and (high - low) >= self.min_contrast
        # Otsu will split even tiny variations in plain paper. Skip it for
        # solid/low-contrast fills, including black, and retain the true colour.
        if has_pattern:
            _, pattern_mask = cv2.threshold(
                gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
            )
        else:
            pattern_mask = np.zeros(gray.shape, dtype=np.uint8)

        background_pixels = img[pattern_mask == 0]
        if not background_pixels.size:
            # A usable background cannot be inferred from an all-ink mask.
            # Treat it as a fill rather than taking the median of no pixels.
            pattern_mask = np.zeros(gray.shape, dtype=np.uint8)
            background_pixels = img.reshape(-1, 3)
            has_pattern = False

        median_colour = np.median(
            background_pixels,
            axis=0
        ).astype(np.uint8)

        pattern_density = np.mean(pattern_mask > 0)

        fft_features = self._extract_fft(
            pattern_mask
        )

        return {
            "median_colour": median_colour,
            "pattern_density": pattern_density,
            "fft_features": fft_features,
            "pattern_mask": pattern_mask,
            "has_pattern": bool(has_pattern),
        }

    def _extract_fft(self, pattern_mask):
        # Pool frequency energy instead of resizing the spectrum: resizing
        # can miss narrow peaks entirely and make different patterns identical.
        return pooled_fft(pattern_mask)

    def build_vector(self, features):
        """28 numbers: 3 normalised BGR colours + density + 24 texture values."""
        texture = np.asarray(features["fft_features"])
        if texture.shape != (24,) or not np.isfinite(texture).all():
            raise ValueError("Expected 24 pooled FFT values; recompute old saved features.")
        return np.concatenate([
            features["median_colour"] / 255.0,
            [features["pattern_density"]],
            texture
        ])

def bgr_to_lab(bgr):
    """OpenCV 8-bit Lab -> real Lab (L 0-100, a/b around 0), so Delta-E is in standard units."""
    L, a, b = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2LAB)[0, 0].astype(float)
    return np.array([L * 100 / 255, a - 128, b - 128])


_grid = {}


def pooled_fft(mask, nr=6, na=4):
    """FFT texture pooled into spacing (radial) x orientation (angle) bins.

    Used for both interval patches and legend swatches. This descriptor:
      - averages over 32-row tiles, so tall intervals aren't one blurry spectrum
      - applies a Hann window, so tile edges don't add fake frequencies
      - pools into 24 bins in cycles/pixel, which mean the same for any patch size
    A solid mask has no texture and returns zeros.
    """
    m = mask.astype(np.float32)
    if m.ndim != 2 or m.size == 0 or not np.isfinite(m).all():
        raise ValueError("Expected a non-empty, finite 2-D pattern mask.")
    if not isinstance(nr, int) or not isinstance(na, int) or nr < 1 or na < 1:
        raise ValueError("FFT bin counts must be positive integers.")
    h, w = m.shape
    th = 32 if h >= 32 else h
    specs = []
    for y in range(0, h - th + 1, th):
        t = m[y:y + th]
        key = (th, w, nr, na)
        if key not in _grid:
            fy = (np.arange(th) - th // 2) / th
            fx = (np.arange(w) - w // 2) / w
            FY, FX = np.meshgrid(fy, fx, indexing="ij")
            rb = np.minimum((np.hypot(FY, FX) / 0.5 * nr).astype(int), nr - 1)
            ab = np.minimum(((np.arctan2(FY, FX) % np.pi) / np.pi * na).astype(int), na - 1)
            _grid[key] = ((rb * na + ab).ravel(), np.outer(np.hanning(th), np.hanning(w)))
        bins, hann = _grid[key]
        S = np.abs(np.fft.fftshift(np.fft.fft2((t - t.mean()) * hann)))
        specs.append(np.bincount(bins, weights=S.ravel(), minlength=nr * na))
    f = np.mean(specs, axis=0)
    s = f.sum()
    return f / s if s > 0 else f
