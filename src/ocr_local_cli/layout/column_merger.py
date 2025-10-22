from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np

from ..ocr.base import OCRToken


@dataclass
class LayoutMetadata:
    columns: int
    notes: List[str]


def sort_tokens_reading_order(tokens: Sequence[OCRToken]) -> List[OCRToken]:
    """
    Reorder OCR tokens using a heuristic for multi-column resumes.
    """
    if not tokens:
        return []

    centers = np.array([_bbox_center(token.bbox) for token in tokens])
    x_coords = centers[:, 0]
    width = x_coords.max() - x_coords.min()
    if width == 0:
        width = 1.0
    normalized_x = (x_coords - x_coords.min()) / width
    column_break = 0.5
    left = []
    right = []
    for token, x in zip(tokens, normalized_x):
        if x < column_break:
            left.append(token)
        else:
            right.append(token)

    if not left or not right:
        # Single column assumption
        ordered = sorted(tokens, key=lambda t: (_bbox_center(t.bbox)[1], _bbox_center(t.bbox)[0]))
        return ordered

    left_sorted = sorted(left, key=lambda t: (_bbox_center(t.bbox)[1], _bbox_center(t.bbox)[0]))
    right_sorted = sorted(right, key=lambda t: (_bbox_center(t.bbox)[1], _bbox_center(t.bbox)[0]))
    # Merge columns by zipping line by line
    merged: List[OCRToken] = []
    while left_sorted or right_sorted:
        if left_sorted:
            merged.append(left_sorted.pop(0))
        if right_sorted:
            merged.append(right_sorted.pop(0))
    return merged


def analyze_layout(tokens: Sequence[OCRToken]) -> LayoutMetadata:
    if not tokens:
        return LayoutMetadata(columns=0, notes=["no tokens"])

    centers = np.array([_bbox_center(token.bbox) for token in tokens])
    xs = centers[:, 0]
    width = xs.max() - xs.min()
    if width == 0:
        return LayoutMetadata(columns=1, notes=["uniform x distribution"])

    histogram, _ = np.histogram(xs, bins=4)
    empty_bins = (histogram == 0).sum()
    columns = 1
    notes: List[str] = []
    if empty_bins >= 1:
        columns = 2
        notes.append("detected potential column gap")
    return LayoutMetadata(columns=columns, notes=notes)


def _bbox_center(bbox):
    # BBox is list of 4 tuples (x, y)
    xs = [point[0] for point in bbox]
    ys = [point[1] for point in bbox]
    return (sum(xs) / len(xs), sum(ys) / len(ys))

