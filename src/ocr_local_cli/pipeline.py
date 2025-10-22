from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import cv2

from .config import PipelineConfig, load_config, load_json_schema
from .extraction.llm_client import LLMClient
from .extraction.parser import ExtractionResult, ResumeExtractor
from .layout.column_merger import LayoutMetadata, analyze_layout, sort_tokens_reading_order
from .ocr.base import BaseOCREngine, OCRToken
from .ocr.paddle_client import PaddleOCREngine
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
        self.ocr_engine = ocr_engine or PaddleOCREngine(
            lang=self.config.ocr.lang,
            use_gpu=self.config.ocr.use_gpu,
            enable_angle_cls=self.config.ocr.enable_angle_cls,
            cls_batch_num=self.config.ocr.cls_batch_num,
            rec_batch_num=self.config.ocr.rec_batch_num,
        )
        self.llm_client = llm_client or LLMClient(self.config.llm)
        schema_dict = load_json_schema(self.config.schema_path)
        taxonomy = self._load_skill_taxonomy(self.config.skill_taxonomy_path)
        self.extractor = ResumeExtractor(self.config, self.llm_client, schema_dict, taxonomy)

    def run(self, path: Path) -> PipelineOutput:
        logger.info("Processing %s", path)
        image_paths = self._prepare_images(path)
        tokens = list(self.ocr_engine.recognize(image_paths))
        logger.info("OCR produced %d tokens", len(tokens))
        high_conf_tokens = self._filter_tokens(tokens)
        layout = analyze_layout(high_conf_tokens)
        if self.config.enable_layout_analysis:
            ordered = sort_tokens_reading_order(high_conf_tokens)
        else:
            ordered = sorted(high_conf_tokens, key=lambda t: (t.page, t.bbox[0][1], t.bbox[0][0]))
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
            cv2.imwrite(str(tmp_path), preprocess_result.image)
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

