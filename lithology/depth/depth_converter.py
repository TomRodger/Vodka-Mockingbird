"""Take teammates' layer boundaries in pixels and add their actual depths.

extract_depths.py reads the scan's ruler first. Each marker links an image row
to a printed depth. This file uses that ruler to find each layer's top and base.
"""

import argparse
import csv
from bisect import bisect_right
from math import isfinite
from pathlib import Path


class DepthConverter:
    """Store one scan's ruler so we can use it for every layer in that scan."""

    def __init__(self, markers, max_extrapolation_pixels=0):
        # markers contains (pixel_y, depth) pairs from extract_depths.py.
        # Normally we only calculate between known labels. A caller can choose
        # a small pixel allowance beyond the first/last label if needed.
        self.max_extrapolation_pixels = float(max_extrapolation_pixels)
        if not isfinite(self.max_extrapolation_pixels) or self.max_extrapolation_pixels < 0:
            raise ValueError("Extrapolation limit must be a finite non-negative number.")
        # Put ruler labels in order from the top of the image to the bottom.
        self.markers = sorted((float(y), float(depth)) for y, depth in markers)
        if len(self.markers) < 2:
            raise ValueError("Provide at least two depth markers.")
        if not all(isfinite(v) for pair in self.markers for v in pair):
            raise ValueError("Marker positions and depths must be finite numbers.")
        # Going down the image should also mean going deeper into the ground.
        for (y1, d1), (y2, d2) in zip(self.markers, self.markers[1:]):
            if y2 <= y1 or d2 <= d1:
                raise ValueError("Pixel positions and depths must both increase.")
        self.positions = [y for y, _ in self.markers]

    def depth_at(self, y):
        """Find the depth at one image row using the nearest ruler labels."""
        y = float(y)
        if not isfinite(y) or not self.positions[0] - self.max_extrapolation_pixels <= y <= self.positions[-1] + self.max_extrapolation_pixels:
            raise ValueError(f"Row {y} is outside the calibrated depth range.")
        # Find the pair of labels around this row. At an allowed edge extension,
        # use the first or last pair instead.
        index = max(0, min(bisect_right(self.positions, y) - 1, len(self.markers) - 2))
        y1, d1 = self.markers[index]
        y2, d2 = self.markers[index + 1]
        # Halfway between two label positions means halfway between their depths.
        return d1 + (y - y1) * (d2 - d1) / (y2 - y1)

    def convert_interval(self, interval):
        """Take one layer with top_y/bottom_y and return it with depths added."""
        top_y, bottom_y = float(interval["top_y"]), float(interval["bottom_y"])
        if bottom_y <= top_y:
            raise ValueError("bottom_y must be greater than top_y.")
        # Copy the row so teammates' input is untouched. Keep any lithology or
        # cluster fields they have already added.
        result = dict(interval)
        result["top_depth"] = self.depth_at(top_y)
        result["base_depth"] = self.depth_at(bottom_y)
        if self.max_extrapolation_pixels:
            # Tell the caller when a result extends beyond readable labels.
            result["depth_extrapolated"] = top_y < self.positions[0] or bottom_y > self.positions[-1]
        # Some code calls the base 'bottom_depth'. Update that name too if used.
        if "bottom_depth" in result:
            result["bottom_depth"] = result["base_depth"]
        return result


def main():
    """Allow the converter to be run from the terminal with CSV files."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--markers", type=Path, help="CSV with pixel_y,depth columns")
    parser.add_argument("--intervals", type=Path, help="CSV with top_y,bottom_y columns")
    parser.add_argument("--output", type=Path, help="Destination CSV (must be a new file)")
    parser.add_argument("--max-extrapolation-pixels", type=float, default=0,
                        help="Allow this many pixels beyond known markers; default 0")
    args = parser.parse_args()
    if not all((args.markers, args.intervals, args.output)):
        parser.error("Provide --markers, --intervals and --output.")
    try:
        # Read the ruler once, then use it to convert every layer in the CSV.
        with args.markers.open(newline="", encoding="utf-8-sig") as stream:
            converter = DepthConverter(((r["pixel_y"], r["depth"]) for r in csv.DictReader(stream)), args.max_extrapolation_pixels)
        with args.intervals.open(newline="", encoding="utf-8-sig") as stream:
            rows = [converter.convert_interval(r) for r in csv.DictReader(stream)]
        if not rows:
            raise ValueError("The intervals CSV contains no intervals.")
        # 'x' creates a new file and prevents overwriting an existing result.
        with args.output.open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {len(rows)} intervals to {args.output}")
    except (ValueError, KeyError, OSError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
