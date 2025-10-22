from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, List, Sequence

from .base import BaseOCREngine, OCRToken

logger = logging.getLogger("ocr_local_cli.ocr.paddle")

try:
    from paddleocr import PaddleOCR  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    PaddleOCR = None  # type: ignore


@dataclass
class PaddleOCREngine(BaseOCREngine):
    name: str = "paddleocr"
    lang: str = "en"
    use_gpu: bool = False
    enable_angle_cls: bool = True
    cls_batch_num: int = 30
    rec_batch_num: int = 6

    def __post_init__(self) -> None:
        if PaddleOCR is None:
            raise ImportError(
                "PaddleOCR is not available. Install paddleocr to use this backend."
            )
        self._client = PaddleOCR(
            use_angle_cls=self.enable_angle_cls,
            lang=self.lang,
            use_gpu=self.use_gpu,
            cls_batch_num=self.cls_batch_num,
            rec_batch_num=self.rec_batch_num,
        )

    def recognize(self, images: Sequence[str]) -> Iterable[OCRToken]:
        logger.info("Running PaddleOCR on %d pages", len(images))
        for page_num, image_path in enumerate(images):
            results = self._client.ocr(image_path, cls=True)
            for result in results:
                bbox = result[0]
                text, conf = result[1][0], float(result[1][1])
                yield OCRToken(text=text, confidence=conf, bbox=bbox, page=page_num)

