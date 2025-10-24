from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from .config import PipelineConfig, load_config, load_json_schema
from .extraction.llm_client import LLMClient
from .extraction.parser import ExtractionResult, ResumeExtractor
from .layout.column_merger import LayoutMetadata, analyze_layout, sort_tokens_reading_order
from .ocr.base import BaseOCREngine, OCRToken
from .ocr.tesseract_client import TesseractOCREngine
from .preprocessing.image_ops import preprocess_for_ocr
from .utils.files import iter_image_paths
from .utils.logging import get_logger

logger = get_logger("pipeline")


@dataclass
class PipelineOutput:
    document: dict
    raw_response: dict
    prompt: str
    used_fallback: bool
    tokens: List[OCRToken]
    layout: LayoutMetadata


class ResumeOCRPipeline:
    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        ocr_engine: Optional[BaseOCREngine] = None,
        llm_client: Optional[LLMClient] = None,
    ) -> None:
        self.config = config or load_config()
        self.config.tmp_dir.mkdir(parents=True, exist_ok=True)
        self.ocr_engine = ocr_engine or self._build_ocr_engine()
        self.llm_client = llm_client or LLMClient(self.config.llm)
        schema_dict = load_json_schema(self.config.schema_path)
        taxonomy = self._load_skill_taxonomy(self.config.skill_taxonomy_path)
        self.extractor = ResumeExtractor(self.config, self.llm_client, schema_dict, taxonomy)

    def run(self, path: Path) -> PipelineOutput:
        logger.info("Processing %s", path)
        image_paths = self._prepare_images(path)
        tokens = list(self.ocr_engine.recognize(image_paths))
        tokens = [token for token in tokens if token.text]
        logger.info("OCR produced %d tokens", len(tokens))
        for idx, token in enumerate(tokens):
            logger.debug(
                "OCR Token %03d | page=%d | conf=%.2f | text=%s",
                idx,
                token.page,
                token.confidence,
                token.text,
            )
        high_conf_tokens = self._filter_tokens(tokens)
        if not high_conf_tokens:
            logger.warning("No OCR tokens passed confidence threshold; aborting extraction.")
            layout = LayoutMetadata(columns=0, notes=["no tokens"])
            extraction = self.extractor.extract([], layout.notes)
            return PipelineOutput(
                document=extraction.document.ordered(),
                raw_response=extraction.raw_response,
                prompt=extraction.prompt,
                used_fallback=extraction.used_fallback,
                tokens=[],
                layout=layout,
            )
        layout = analyze_layout(high_conf_tokens)
        logger.debug("Layout analysis notes: %s", layout.notes)
        if self.config.enable_layout_analysis:
            ordered = sort_tokens_reading_order(high_conf_tokens)
        else:
            ordered = sorted(high_conf_tokens, key=lambda t: (t.page, t.bbox[0][1], t.bbox[0][0]))
        logger.debug("Ordered %d tokens for extraction", len(ordered))
        extraction = self.extractor.extract(ordered, layout.notes)
        return PipelineOutput(
            document=extraction.document.ordered(),
            raw_response=extraction.raw_response,
            prompt=extraction.prompt,
            used_fallback=extraction.used_fallback,
            tokens=ordered,
            layout=layout,
        )

    def run_batch(self, paths: Iterable[Path]) -> List[PipelineOutput]:
        return [self.run(path) for path in paths]

    def _prepare_images(self, path: Path) -> List[str]:
        prepared_paths: List[str] = []
        for idx, image_path in enumerate(iter_image_paths(path)):
            preprocess_result = preprocess_for_ocr(image_path)
            tmp_path = self.config.tmp_dir / f"{image_path.stem}_processed_{idx:03d}.png"
            preprocess_result.image.save(str(tmp_path))
            prepared_paths.append(str(tmp_path))
        if not prepared_paths:
            raise FileNotFoundError(f"No images found for {path}")
        return prepared_paths

    def _filter_tokens(self, tokens: Sequence[OCRToken]) -> List[OCRToken]:
        filtered = [token for token in tokens if token.confidence >= self.config.min_confidence]
        if not filtered:
            logger.warning(
                "No tokens met min confidence %.2f, using fallback threshold %.2f",
                self.config.min_confidence,
                self.config.fallback_confidence,
            )
            filtered = [
                token for token in tokens if token.confidence >= self.config.fallback_confidence
            ]
        return filtered

    def _build_ocr_engine(self) -> BaseOCREngine:
        engine = (self.config.ocr.engine or "paddle").lower()
        if engine == "paddle":
            try:
                from .ocr.paddle_client import PaddleOCREngine  # local import

                return PaddleOCREngine(
                    lang=self.config.ocr.lang,
                    enable_angle_cls=self.config.ocr.enable_angle_cls,
                    cls_batch_num=self.config.ocr.cls_batch_num,
                    rec_batch_num=self.config.ocr.rec_batch_num,
                    model_root=Path(self.config.ocr.model_root).expanduser()
                    if self.config.ocr.model_root
                    else None,
                    cpu_threads=self.config.ocr.cpu_threads,
                    use_gpu=self.config.ocr.use_gpu,
                    gpu_id=self.config.ocr.gpu_id,
                    model_variant=self.config.ocr.model_variant,
                    use_doc_orientation_classify=self.config.ocr.use_doc_orientation_classify,
                    use_doc_unwarping=self.config.ocr.use_doc_unwarping,
                )
            except Exception as exc:  # pragma: no cover - fallback path
                logger.warning(
                    "Failed to initialize PaddleOCR (%s). Falling back to Tesseract.", exc
                )
                return TesseractOCREngine(lang=self._tesseract_lang())
        if engine == "tesseract":
            return TesseractOCREngine(lang=self._tesseract_lang())
        logger.warning("Unknown OCR engine '%s'. Falling back to Tesseract.", engine)
        return TesseractOCREngine(lang=self._tesseract_lang())

    def _tesseract_lang(self) -> str:
        lang = self.config.ocr.lang
        if lang in ("en", "eng"):
            return "eng"
        return lang or "eng"

    def _load_skill_taxonomy(self, path: Optional[Path]) -> Optional[List[str]]:
        if not path:
            return None
        full_path = path.expanduser()
        if not full_path.exists():
            logger.warning("Skill taxonomy file not found: %s", full_path)
            return None
        with full_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, list):
            return [str(item) for item in data]
        if isinstance(data, dict) and "skills" in data:
            return [str(item) for item in data["skills"]]
        logger.warning("Skill taxonomy format not recognized. Expected list or {skills: []}.")
        return None
