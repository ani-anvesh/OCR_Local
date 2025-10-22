from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np
from PIL import Image

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
    enable_angle_cls: bool = True
    cls_batch_num: int = 30
    rec_batch_num: int = 6
    model_root: Optional[Path] = None
    cpu_threads: int = 1

    def __post_init__(self) -> None:
        if PaddleOCR is None:
            raise ImportError(
                "PaddleOCR is not available. Install paddleocr to use this backend."
            )
        import os
        import paddle

        os.environ.setdefault("PADDLEX_OFFLINE_MODE", "1")
        os.environ.setdefault("OMP_NUM_THREADS", str(max(1, self.cpu_threads)))
        os.environ.setdefault("MKL_NUM_THREADS", str(max(1, self.cpu_threads)))
        os.environ.setdefault("MKL_SERVICE_FORCE_INTEL", "1")
        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")
        paddle.device.set_device("cpu")
        model_kwargs = self._model_dirs()
        self._client = PaddleOCR(
            use_angle_cls=self.enable_angle_cls,
            lang=self.lang,
            cls_batch_num=self.cls_batch_num,
            rec_batch_num=self.rec_batch_num,
            cpu_threads=self.cpu_threads,
            **model_kwargs,
        )

    def recognize(self, images: Sequence[str]) -> Iterable[OCRToken]:
        logger.info("Running PaddleOCR on %d pages", len(images))
        for page_num, image_path in enumerate(images):
            pil_image = Image.open(image_path).convert("RGB")
            ndarray = np.array(pil_image)
            try:
                results = self._client.ocr(ndarray)
            except Exception:
                results = self._client.ocr(image_path)
            for result in results:
                bbox, text, conf = self._extract_entry(result)
                if not text:
                    logger.debug("Skipping empty text entry: result=%s", result)
                    continue
                yield OCRToken(text=text, confidence=conf, bbox=bbox, page=page_num)

    def _extract_entry(self, result):
        bbox = []
        text = ""
        confidence = 0.0

        if isinstance(result, (list, tuple)):
            if len(result) >= 2 and PaddleOCREngine._looks_like_bbox(result[0]):
                bbox = self._normalize_bbox(result[0])
                text, confidence = self._parse_info_block(result[1])
            elif len(result) == 2:
                bbox = self._normalize_bbox(result[0])
                text, confidence = self._parse_info_block(result[1])
            else:
                text, confidence = self._parse_info_block(result)
        elif isinstance(result, dict):
            bbox = self._normalize_bbox(
                result.get("box") or result.get("bbox") or result.get("points") or []
            )
            text, confidence = self._parse_info_block(result)
        else:
            text, confidence = self._parse_info_block(result)

        return bbox, text, confidence

    def _parse_info_block(self, info):
        text = ""
        confidence = 0.0
        if isinstance(info, (list, tuple)):
            for element in info:
                if isinstance(element, str) and not text:
                    text = element
                elif isinstance(element, (list, tuple, dict)) and not text:
                    nested_text, nested_conf = self._parse_info_block(element)
                    if nested_text and not text:
                        text = nested_text
                    if nested_conf and not confidence:
                        confidence = nested_conf
                else:
                    value = self._to_float(element)
                    if value and not confidence:
                        confidence = value
        elif isinstance(info, dict):
            text = (
                info.get("text")
                or info.get("value")
                or info.get("transcription")
                or info.get("rec_text")
                or ""
            )
            confidence = self._to_float(
                info.get("score")
                or info.get("confidence")
                or info.get("probability")
                or 0.0
            )
        elif isinstance(info, str):
            text = info
        elif isinstance(info, (int, float)):
            confidence = float(info)
        return text.strip(), confidence

    @staticmethod
    def _to_float(value):
        if isinstance(value, (list, tuple)):
            for element in value:
                try:
                    return float(element)
                except (TypeError, ValueError):
                    continue
            return 0.0
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _looks_like_bbox(obj) -> bool:
        if not isinstance(obj, (list, tuple)):
            return False
        if obj and isinstance(obj[0], (list, tuple)) and len(obj[0]) == 2:
            return True
        return False

    @staticmethod
    def _normalize_bbox(bbox):
        if bbox is None:
            return []
        if isinstance(bbox, (list, tuple)):
            if len(bbox) == 1 and isinstance(bbox[0], (list, tuple)):
                return PaddleOCREngine._normalize_bbox(bbox[0])
            if all(isinstance(pt, (list, tuple)) and len(pt) == 2 for pt in bbox):
                return [(float(pt[0]), float(pt[1])) for pt in bbox]
            flat = []
            for item in bbox:
                if isinstance(item, (int, float)):
                    flat.append(float(item))
            if flat and len(flat) % 2 == 0:
                return [(flat[i], flat[i + 1]) for i in range(0, len(flat), 2)]
        return []

    def _model_dirs(self) -> dict:
        """
        Build explicit model directory arguments to avoid repeated network checks.
        """
        root = self.model_root or Path.home() / ".paddlex" / "official_models"
        det_candidates = [
            root / "PP-OCRv5_server_det",
            root / "en_PP-OCRv3_det",
            root / "en_PP-OCRv4_det",
        ]
        rec_candidates = [
            root / "en_PP-OCRv5_mobile_rec",
            root / "en_PP-OCRv3_rec",
            root / "en_PP-OCRv4_rec",
        ]
        cls_candidates = [
            root / "PP-LCNet_x1_0_textline_ori",
            root / "PP-LCNet_x1_0_doc_ori",
            root / "PP-LCNet_x1_0_cls",
            root / "ch_ppocr_mobile_v2.0_cls",
        ]
        kwargs = {}
        for candidate in det_candidates:
            if candidate.exists():
                kwargs["det_model_dir"] = str(candidate)
                break
        for candidate in rec_candidates:
            if candidate.exists():
                kwargs["rec_model_dir"] = str(candidate)
                break
        for candidate in cls_candidates:
            if candidate.exists():
                kwargs["cls_model_dir"] = str(candidate)
                break
        return kwargs
