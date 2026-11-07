"""
Texture descriptors used by texture_test.py to compare FFT vs HOG variants.
(Separate from lithology/features.py, which is what the pipeline uses.)

Every function takes one window: an H x W x 3 float array in [0, 1] (RGB),
cut from inside a single interval. Windows can be any size, including
non-square ones as wide as a narrow column.

    colour_feat      median fill colour in Lab (3 numbers)
    fft_feat         FFT magnitude pooled into spacing x orientation bins
    hog_pooled_feat  HOG averaged over position: global edge-direction histogram
    hog_concat_feat  standard HOG, per-cell histograms concatenated

Texture features are computed on binarised black linework only, so fill
colour never leaks into texture.
"""

import numpy as np
from skimage.color import rgb2lab
from skimage.feature import hog

INK_MARGIN = 0.2    # linework must be this much darker than the fill (light fills)
INK_RATIO = 0.6     # ...or darker than this fraction of the fill (dark fills)


def to_gray(win):
    return win.mean(axis=2) if win.ndim == 3 else win


def binarise(win):
    """1 where a pixel is pattern linework, 0 for fill.

    The threshold is relative to the window's fill (its median grey), because
    documents differ: most draw black lines, but some (e.g. FREEMAN) draw grey
    lines (~0.65) on a pale fill, and some use dark fills like dark green.
      light fill: ink = darker than fill - INK_MARGIN
      dark fill:  ink = darker than INK_RATIO x fill
    A fixed "black" threshold sees no ink at all in the grey-line documents.
    """
    g = to_gray(win)
    m = float(np.median(g))
    t = max(INK_RATIO * m, m - INK_MARGIN)
    return (g < t).astype(float)


def colour_feat(win):
    """Median Lab colour of the FILL. Lab distance roughly matches perceived difference.

    Fill = the brightest quarter of the pixels: linework is always the darkest part of a
    pattern. A threshold relative to the median (or the brightest half) fails when
    linework covers most of a small window: on the example some 16x16 windows are over
    half black, so the 'fill' came out black and 'texture + colour' scored worse than
    texture alone at 16 px."""
    lab = rgb2lab(win).reshape(-1, 3)
    g = to_gray(win).reshape(-1)
    return np.median(lab[g >= np.percentile(g, 75)], axis=0)


# ---------------------------------------------------------------- FFT

_grid_cache = {}


def _fft_grid(h, w, nr, na):
    key = (h, w, nr, na)
    if key not in _grid_cache:
        fy = (np.arange(h) - h // 2) / h       # cycles per pixel, matches fftshift
        fx = (np.arange(w) - w // 2) / w
        FY, FX = np.meshgrid(fy, fx, indexing="ij")
        r = np.hypot(FY, FX) / 0.5             # 1.0 = finest possible pattern
        a = np.arctan2(FY, FX) % np.pi         # magnitude is symmetric: fold to [0, pi)
        rb = np.minimum((r * nr).astype(int), nr - 1)
        ab = np.minimum((a / np.pi * na).astype(int), na - 1)
        bins = (rb * na + ab).ravel()
        hann = np.outer(np.hanning(h), np.hanning(w))
        _grid_cache[key] = (bins, hann)
    return _grid_cache[key]


def fft_feat(win, nr=6, na=4):
    """Texture as 'how fine' (radial bins) x 'which direction' (angle bins).

    Bins are in cycles per pixel, so they mean the same thing for any window
    size. The magnitude ignores where the pattern starts (phase); the Hann
    window stops the window's own edges adding fake frequencies.
    """
    ink = binarise(win)
    h, w = ink.shape
    bins, hann = _fft_grid(h, w, nr, na)
    S = np.abs(np.fft.fftshift(np.fft.fft2((ink - ink.mean()) * hann)))
    f = np.bincount(bins, weights=S.ravel(), minlength=nr * na)
    s = f.sum()
    return f / s if s > 0 else f


# ---------------------------------------------------------------- HOG

def _hog(win, flat):
    return hog(binarise(win), orientations=9, pixels_per_cell=(8, 8),
               cells_per_block=(2, 2), feature_vector=flat)


def hog_concat_feat(win):
    """Standard HOG. Position-dependent: shifting the pattern changes it."""
    return _hog(win, True)


def hog_pooled_feat(win):
    """HOG averaged over all positions: which edge directions occur, not where."""
    return _hog(win, False).mean(axis=(0, 1)).ravel()


TEXTURE_DESCRIPTORS = {
    "fft": fft_feat,
    "hog_pooled": hog_pooled_feat,
    "hog_concat": hog_concat_feat,
}
