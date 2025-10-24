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
    use_gpu: bool = False
    gpu_id: int = 0

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
        if self.use_gpu and paddle.device.is_compiled_with_cuda():
            device = f"gpu:{self.gpu_id}"
        else:
            if self.use_gpu and not paddle.device.is_compiled_with_cuda():
                logger.warning("PaddlePaddle is not compiled with CUDA; reverting to CPU")
            device = "cpu"
        paddle.device.set_device(device)
        model_kwargs = self._model_dirs()
        self._client = PaddleOCR(
            lang=self.lang,
            use_textline_orientation=self.enable_angle_cls,
            textline_orientation_batch_size=self.cls_batch_num,
            text_recognition_batch_size=self.rec_batch_num,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            cpu_threads=self.cpu_threads,
            device=device,
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
                if isinstance(result, dict) and result.get("rec_texts"):
                    yield from self._yield_dict_tokens(result, page_num)
                    continue
                bbox, text, conf = self._extract_entry(result)
                if not text:
                    logger.debug("Skipping empty text entry: result=%s", result)
                    continue
                yield OCRToken(text=text, confidence=conf, bbox=bbox, page=page_num)

    def _yield_dict_tokens(self, result, page_num):
        texts = result.get("rec_texts") or []
        if not isinstance(texts, (list, tuple)):
            return
        scores = result.get("rec_scores") or result.get("scores") or []
        boxes = (
            result.get("rec_polys")
            or result.get("rec_boxes")
            or result.get("dt_polys")
            or []
        )
        for idx, raw_text in enumerate(texts):
            text = (raw_text or "").strip()
            if not text:
                continue
            bbox = self._normalize_bbox(self._select_box(boxes, idx))
            confidence = self._select_score(scores, idx)
            yield OCRToken(text=text, confidence=confidence, bbox=bbox, page=page_num)

    def _select_box(self, boxes, idx):
        if boxes is None:
            return []
        import numpy as np

        if isinstance(boxes, np.ndarray):
            if boxes.ndim == 3 and idx < boxes.shape[0]:
                return boxes[idx]
            if boxes.ndim == 2:
                if idx < boxes.shape[0]:
                    return boxes[idx]
                return boxes
            return []
        if isinstance(boxes, (list, tuple)):
            if idx < len(boxes):
                return boxes[idx]
        return []

    def _select_score(self, scores, idx):
        if scores is None:
            return 0.0
        import numpy as np

        if isinstance(scores, np.ndarray):
            if scores.ndim == 1 and idx < scores.shape[0]:
                return float(scores[idx])
            if scores.ndim > 1 and idx < scores.shape[0]:
                return float(scores[idx][0])
            return 0.0
        if isinstance(scores, (list, tuple)) and idx < len(scores):
            try:
                return float(scores[idx])
            except (TypeError, ValueError):
                return 0.0
        return 0.0

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
                result.get("box")
                or result.get("bbox")
                or result.get("points")
                or result.get("rec_polys")
                or result.get("rec_boxes")
                or result.get("dt_polys")
                or []
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
            if not text and info.get("rec_texts"):
                first_text = info.get("rec_texts")
                if isinstance(first_text, (list, tuple)) and first_text:
                    text = str(first_text[0])
                elif isinstance(first_text, str):
                    text = first_text
            confidence = self._to_float(
                info.get("score")
                or info.get("confidence")
                or info.get("probability")
                or 0.0
            )
            if not confidence:
                rec_scores = info.get("rec_scores")
                if isinstance(rec_scores, (list, tuple)) and rec_scores:
                    confidence = self._to_float(rec_scores[0])
                else:
                    try:
                        import numpy as np

                        if isinstance(rec_scores, np.ndarray) and rec_scores.size:
                            confidence = float(rec_scores.flat[0])
                    except ImportError:  # pragma: no cover
                        pass
        elif isinstance(info, str):
            text = info
        elif isinstance(info, (int, float)):
            confidence = float(info)
        return text.strip(), confidence

    @staticmethod
    def _to_float(value):
        try:
            import numpy as np
        except ImportError:  # pragma: no cover
            np = None
        if np is not None and isinstance(value, np.ndarray):
            if value.size:
                return float(value.flat[0])
            return 0.0
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
        try:
            import numpy as np
        except ImportError:  # pragma: no cover
            np = None
        if np is not None and isinstance(bbox, np.ndarray):
            bbox = bbox.tolist()
        if isinstance(bbox, (list, tuple)):
            if not bbox:
                return []
            first = bbox[0]
            if np is not None and isinstance(first, np.ndarray):
                bbox = [item.tolist() if isinstance(item, np.ndarray) else item for item in bbox]
                first = bbox[0] if bbox else []
            if len(bbox) == 1 and isinstance(first, (list, tuple)) and len(bbox) == 1:
                return PaddleOCREngine._normalize_bbox(first)
            if all(isinstance(pt, (list, tuple)) and len(pt) == 2 for pt in bbox):
                return [(float(pt[0]), float(pt[1])) for pt in bbox]
            flat = []
            for item in bbox:
                if isinstance(item, (int, float)):
                    flat.append(float(item))
            if flat and len(flat) == 4:
                x1, y1, x2, y2 = flat
                return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
            if flat and len(flat) % 2 == 0:
                return [(flat[i], flat[i + 1]) for i in range(0, len(flat), 2)]
        return []

    def _model_dirs(self) -> dict:
        """
        Build explicit model directory arguments to avoid repeated network checks.
        """
        root = self.model_root or Path.home() / ".paddlex" / "official_models"
        det_candidates = [
            root / "PP-OCRv5_mobile_det",
            root / "en_PP-OCRv3_det",
            root / "en_PP-OCRv4_det",
            root / "PP-OCRv5_server_det",
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
        kwargs = {
            "text_detection_model_name": "PP-OCRv5_mobile_det",
            "text_recognition_model_name": "en_PP-OCRv5_mobile_rec",
        }
        if self.enable_angle_cls:
            kwargs.setdefault("textline_orientation_model_name", "PP-LCNet_x1_0_textline_ori")
        for candidate in det_candidates:
            if candidate.exists():
                kwargs["text_detection_model_dir"] = str(candidate)
                kwargs["text_detection_model_name"] = candidate.name
                break
        for candidate in rec_candidates:
            if candidate.exists():
                kwargs["text_recognition_model_dir"] = str(candidate)
                kwargs["text_recognition_model_name"] = candidate.name
                break
        for candidate in cls_candidates:
            if candidate.exists():
                kwargs["textline_orientation_model_dir"] = str(candidate)
                kwargs.setdefault("textline_orientation_model_name", candidate.name)
                break
        return kwargs
