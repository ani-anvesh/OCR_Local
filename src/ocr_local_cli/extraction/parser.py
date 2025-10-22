from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from ..config import PipelineConfig
from ..ocr.base import OCRToken
from ..schema import ResumeDocument
from ..utils.logging import get_logger
from ..validation import validators
from .llm_client import LLMClient, LLMExtractionError
from .prompt_builder import build_prompt

logger = get_logger("extraction.parser")


@dataclass
class ExtractionResult:
    document: ResumeDocument
    raw_response: Dict
    prompt: str
    used_fallback: bool = False


class ResumeExtractor:
    def __init__(
        self,
        config: PipelineConfig,
        llm_client: LLMClient,
        schema: Dict,
        skill_taxonomy: Optional[List[str]] = None,
    ) -> None:
        self.config = config
        self.llm_client = llm_client
        self.schema = schema
        self.skill_taxonomy = skill_taxonomy

    def extract(self, tokens: Sequence[OCRToken], layout_notes: Optional[List[str]]) -> ExtractionResult:
        prompt = build_prompt(
            self.config.prompt_template_path,
            tokens,
            layout_notes=layout_notes,
            schema=self.schema,
        )
        try:
            response = self.llm_client.extract(prompt)
            document = self._parse_response(response)
            return ExtractionResult(document=document, raw_response=response, prompt=prompt)
        except LLMExtractionError as exc:
            logger.warning("LLM extraction failed, falling back to heuristics: %s", exc)
            document = self._fallback(tokens)
            return ExtractionResult(
                document=document,
                raw_response={"error": str(exc)},
                prompt=prompt,
                used_fallback=True,
            )
        except ValueError as exc:
            logger.warning("Failed to parse LLM response (%s). Using fallback.", exc)
            document = self._fallback(tokens)
            return ExtractionResult(
                document=document,
                raw_response={"error": str(exc)},
                prompt=prompt,
                used_fallback=True,
            )

    def _parse_response(self, response: Dict) -> ResumeDocument:
        if "resume" in response:
            payload = response["resume"]
        else:
            payload = response

        if not isinstance(payload, dict):
            raise ValueError("Response payload is not a dictionary")

        document = ResumeDocument.model_validate(payload)
        document.skills = validators.normalize_skills(document.skills, self.skill_taxonomy)
        document.contact = document.contact.model_copy(
            update=validators.validate_contact_fields(document.contact.model_dump())
        )
        return document

    def _fallback(self, tokens: Sequence[OCRToken]) -> ResumeDocument:
        raw_text = "\n".join(token.text for token in tokens)
        contact = {
            "name": self._guess_name(tokens),
            "email": validators.extract_email(raw_text),
            "phone": validators.extract_phone(raw_text),
        }
        skills = self._guess_skills(raw_text)
        document = ResumeDocument(
            contact=contact,
            skills=skills,
            summary=self._guess_summary(raw_text),
            experience=[],
            education=[],
            extras={"fallback": True},
        )
        return document

    def _guess_name(self, tokens: Sequence[OCRToken]) -> Optional[str]:
        if not tokens:
            return None
        first_line = tokens[0].text.strip()
        if len(first_line.split()) <= 4:
            return first_line
        return None

    def _guess_skills(self, text: str) -> List[str]:
        candidates = []
        if "skills" in text.lower():
            segments = text.lower().split("skills")
            for segment in segments[1:]:
                parts = segment.split("\n", 3)[0]
                tokens = [token.strip().title() for token in re_split_skills(parts)]
                candidates.extend(tokens)
        return validators.normalize_skills(candidates, self.skill_taxonomy)

    def _guess_summary(self, text: str) -> Optional[str]:
        lines = text.splitlines()
        summary_lines = []
        capture = False
        for line in lines:
            if "summary" in line.lower():
                capture = True
                continue
            if capture:
                if line.strip() == "":
                    break
                summary_lines.append(line.strip())
                if len(summary_lines) >= 3:
                    break
        return " ".join(summary_lines) if summary_lines else None


def re_split_skills(segment: str) -> List[str]:
    import re

    return [part for part in re.split(r"[•,;|]", segment) if part]

