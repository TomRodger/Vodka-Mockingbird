"""Shared feature extraction for lithology patches (column intervals and legend swatches).

Input convention: every function takes a BGR uint8 image (cv2.imread order). If you
crop with Pillow (RGB), convert with `arr[:, :, ::-1]` first, otherwise red and blue
swap and colour matches between the column and the legend break silently.

    FeatureExtractor.extract(img) -> pattern mask (Otsu), fill colour, density, 16x16 FFT
    bgr_to_lab(bgr)               -> real Lab colour, where distance ~ perceived difference
    pooled_fft(mask)              -> tiled, windowed FFT pooled into 24 spacing x direction bins
"""

import cv2
import numpy as np


class FeatureExtractor:

    def extract(self, img):

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        _, pattern_mask = cv2.threshold(
            gray,
            0,
            255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        background_pixels = img[pattern_mask == 0]

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
            "pattern_mask": pattern_mask
        }

    def _extract_fft(self, pattern_mask):

        spectrum = np.fft.fft2(
            pattern_mask.astype(np.float32)
        )

        spectrum = np.fft.fftshift(
            spectrum
        )

        magnitude = np.log(
            np.abs(spectrum) + 1
        )

        magnitude = cv2.resize(
            magnitude,
            (16, 16)
        )

        cy = magnitude.shape[0] // 2
        cx = magnitude.shape[1] // 2

        magnitude[
            cy-1:cy+2,
            cx-1:cx+2
        ] = 0

        magnitude /= (
            magnitude.max() + 1e-6
        )

        return magnitude.flatten()

    def build_vector(self, features):

        return np.concatenate([
            features["median_colour"] / 255.0,
            [features["pattern_density"]],
            features["fft_features"]
        ])

def bgr_to_lab(bgr):
    """OpenCV 8-bit Lab -> real Lab (L 0-100, a/b around 0), so Delta-E is in standard units."""
    L, a, b = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2LAB)[0, 0].astype(float)
    return np.array([L * 100 / 255, a - 128, b - 128])


_grid = {}


def pooled_fft(mask, nr=6, na=4):
    """FFT texture pooled into spacing (radial) x orientation (angle) bins.

    Computed on FeatureExtractor's pattern mask. Unlike its 16x16 spectrum, this:
      - averages over 32-row tiles, so tall intervals aren't one blurry spectrum
      - applies a Hann window, so tile edges don't add fake frequencies
      - pools into 24 bins in cycles/pixel, which mean the same for any patch size
    On the example the 16x16 spectrum couldn't separate the three rocks; this does.
    """
    m = mask.astype(np.float32)
    h, w = m.shape
    th = 32 if h >= 32 else h
    specs = []
    for y in range(0, h - th + 1, th):
        t = m[y:y + th]
        key = (th, w)
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
