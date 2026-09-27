"""
row_segmentation.py
--------------------
The line-items table has a fixed set of columns but a VARIABLE number of rows
(1 invoice might have 2 items, another might have 8). We can't hardcode row
y-coordinates, so instead we detect rows automatically using a horizontal ink
projection profile: sum up dark ("ink") pixels along each row of a column
strip, and treat contiguous bands of ink as one row, separated by gaps of
blank paper.

Important: we compute the projection using ONLY the "description" column's
interior (away from the ruled table borders). If we included the full table
width, the vertical ruling lines between columns would show up as "ink" in
every row and prevent any gaps from being detected.
"""

from dataclasses import dataclass
from typing import List

import cv2
import numpy as np


@dataclass
class RowBand:
    y_start: int
    y_end: int


def _ink_projection(gray_strip: np.ndarray) -> np.ndarray:
    _, bw = cv2.threshold(gray_strip, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return bw.sum(axis=1) / 255.0


def segment_rows(
    table_bgr: np.ndarray,
    description_col_x_range: tuple,
    ink_threshold: float = 2.0,
    gap_merge_px: int = 25,
    min_band_height: int = 12,
    pad_px: int = 8,
) -> List[RowBand]:
    """
    table_bgr: the full item-table crop (all columns, all rows).
    description_col_x_range: (x0_frac, x1_frac) of the description column,
        as fractions of the table's own width (matches template.json's
        "columns" entries). We shrink it slightly to avoid the column's
        own border lines.

    Returns a list of RowBand(y_start, y_end) in pixel coords relative to
    `table_bgr`, one per detected handwritten row, top to bottom.
    """
    h, w = table_bgr.shape[:2]
    x0_frac, x1_frac = description_col_x_range
    # shrink 5% inward from each side of the column to dodge border ink
    margin = 0.05 * (x1_frac - x0_frac)
    x0 = int((x0_frac + margin) * w)
    x1 = int((x1_frac - margin) * w)
    x0, x1 = max(0, x0), min(w, x1)

    strip = cv2.cvtColor(table_bgr[:, x0:x1], cv2.COLOR_BGR2GRAY)
    proj = _ink_projection(strip)

    has_ink = proj > ink_threshold
    raw_segments = []
    start = None
    for i, v in enumerate(has_ink):
        if v and start is None:
            start = i
        elif not v and start is not None:
            raw_segments.append((start, i))
            start = None
    if start is not None:
        raw_segments.append((start, len(has_ink)))

    # drop specks (stray marks / border bleed) that are too thin to be real text
    raw_segments = [s for s in raw_segments if (s[1] - s[0]) >= 4]

    # merge segments that are close together (multi-line wrapped text within one row)
    merged: List[list] = []
    for s in raw_segments:
        if merged and s[0] - merged[-1][1] < gap_merge_px:
            merged[-1][1] = s[1]
        else:
            merged.append(list(s))

    bands = [
        RowBand(max(0, s - pad_px), min(h, e + pad_px))
        for s, e in merged
        if (e - s) >= min_band_height
    ]
    return bands
