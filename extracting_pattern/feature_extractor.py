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