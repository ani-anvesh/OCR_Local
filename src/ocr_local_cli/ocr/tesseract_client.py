from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from PIL import Image

import pytesseract

from .base import BaseOCREngine, OCRToken


@dataclass
class TesseractOCREngine(BaseOCREngine):
    """Simple OCR engine powered by pytesseract."""

    name: str = "tesseract"
    lang: str = "eng"

    def recognize(self, images: Sequence[str]) -> Iterable[OCRToken]:
        for page_num, image_path in enumerate(images):
            image = Image.open(image_path)
            data = pytesseract.image_to_data(
                image, lang=self.lang, output_type=pytesseract.Output.DICT
            )
            for idx in range(len(data["text"])):
                text = data["text"][idx].strip()
                if not text:
                    continue
                conf_str = data["conf"][idx]
                try:
                    confidence = float(conf_str) / 100.0
                except (ValueError, TypeError):
                    confidence = 0.0
                left = data["left"][idx]
                top = data["top"][idx]
                width = data["width"][idx]
                height = data["height"][idx]
                bbox = [
                    (left, top),
                    (left + width, top),
                    (left + width, top + height),
                    (left, top + height),
                ]
                yield OCRToken(
                    text=text,
                    confidence=confidence,
                    bbox=bbox,
                    page=page_num,
                )

