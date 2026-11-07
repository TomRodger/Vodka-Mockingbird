import cv2
import numpy as np


class LegendExtractor:

    def extract_boxes(self, legend_img):

        gray = cv2.cvtColor(
            legend_img,
            cv2.COLOR_BGR2GRAY
        )

        # Edge detection
        edges = cv2.Canny(
            gray,
            50,
            150
        )

        contours, _ = cv2.findContours(
            edges,
            cv2.RETR_LIST,
            cv2.CHAIN_APPROX_SIMPLE
        )

        results = []

        for contour in contours:

            perimeter = cv2.arcLength(
                contour,
                True
            )

            approx = cv2.approxPolyDP(
                contour,
                0.03 * perimeter,
                True
            )

            # Must look like a rectangle
            if len(approx) != 4:
                continue

            x, y, w, h = cv2.boundingRect(
                approx
            )

            area = w * h

            # Ignore tiny shapes
            if area < 3000:
                continue

            # Ignore whole legend
            if area > (
                legend_img.shape[0]
                * legend_img.shape[1]
                * 0.25
            ):
                continue

            aspect = w / h

            # Your symbol boxes are wider than tall
            if aspect < 1.2:
                continue

            if aspect > 3.5:
                continue

            results.append({
                "bbox": (x, y, w, h),
                "image": legend_img[
                    y:y+h,
                    x:x+w
                ]
            })

        # Remove duplicate rectangles
        filtered = []

        for candidate in results:

            x1, y1, w1, h1 = candidate["bbox"]

            duplicate = False

            for existing in filtered:

                x2, y2, w2, h2 = existing["bbox"]

                if (
                    abs(x1 - x2) < 5 and
                    abs(y1 - y2) < 5 and
                    abs(w1 - w2) < 5 and
                    abs(h1 - h2) < 5
                ):
                    duplicate = True
                    break

            if not duplicate:
                filtered.append(candidate)

        filtered.sort(
            key=lambda b: (
                b["bbox"][1],
                b["bbox"][0]
            )
        )

        return filtered