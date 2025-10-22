from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Protocol, Sequence, Tuple

BoundingBox = List[Tuple[float, float]]


@dataclass
class OCRToken:
    text: str
    confidence: float
    bbox: BoundingBox
    page: int


class BaseOCREngine(Protocol):
    name: str

    def recognize(self, images: Sequence[str]) -> Iterable[OCRToken]:
        """Run OCR on a sequence of image paths."""
        ...

