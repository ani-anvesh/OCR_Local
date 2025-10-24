from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import dateparser

from ..config import PipelineConfig
from ..ocr.base import OCRToken
from ..schema import ResumeDocument, ExperienceItem, EducationItem, ExperienceItem, EducationItem, ExperienceItem, EducationItem
from ..utils.logging import get_logger
from ..validation import validators
from .llm_client import LLMClient, LLMExtractionError
from .prompt_builder import build_prompt, format_tokens_as_text

logger = get_logger("extraction.parser")


@dataclass
class ExtractionResult:
    document: ResumeDocument
    raw_response: Dict
    prompt: str
    raw_text: str
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
        aggregated_text = format_tokens_as_text(tokens)
        prompt = build_prompt(
            self.config.prompt_template_path,
            tokens,
            layout_notes=layout_notes,
            schema=self.schema,
            raw_text=aggregated_text,
        )
        logger.debug("LLM prompt:\n%s", prompt)
        try:
            response = self.llm_client.extract(prompt)
            logger.debug("LLM parsed response: %s", response)
            document = self._parse_response(response, aggregated_text, tokens)
            return ExtractionResult(
                document=document,
                raw_response=response,
                prompt=prompt,
                raw_text=aggregated_text,
            )
        except LLMExtractionError as exc:
            logger.warning("LLM extraction failed, falling back to heuristics: %s", exc)
            document = self._fallback(tokens, aggregated_text)
            return ExtractionResult(
                document=document,
                raw_response={"error": str(exc)},
                prompt=prompt,
                raw_text=aggregated_text,
                used_fallback=True,
            )
        except ValueError as exc:
            logger.warning("Failed to parse LLM response (%s). Using fallback.", exc)
            document = self._fallback(tokens, aggregated_text)
            return ExtractionResult(
                document=document,
                raw_response={"error": str(exc)},
                prompt=prompt,
                raw_text=aggregated_text,
                used_fallback=True,
            )

    def _parse_response(self, response: Dict, raw_text: str, tokens: Sequence[OCRToken]) -> ResumeDocument:
        if "resume" in response:
            payload = response["resume"]
        else:
            payload = response

        if not isinstance(payload, dict):
            raise ValueError("Response payload is not a dictionary")

        allowed_keys = set(ResumeDocument.model_fields.keys())
        unknown_keys = set(payload.keys()) - allowed_keys
        if unknown_keys:
            raise ValueError(f"Response contains unknown fields: {sorted(unknown_keys)}")

        document = ResumeDocument.model_validate(payload)
        document = self._enrich_document(document, raw_text, tokens)
        return document

    def _fallback(self, tokens: Sequence[OCRToken], raw_text: str) -> ResumeDocument:
        contact = {
            "name": self._guess_name(tokens),
            "email": validators.extract_email(raw_text),
            "phone": validators.extract_phone(raw_text),
        }
        logger.debug("Fallback contact info %s", contact)
        skills = self._guess_skills(raw_text)
        logger.debug("Fallback skills detected: %s", skills)
        education, experience = self._extract_structured_entries(raw_text)
        typed_experience = [ExperienceItem(**item) for item in experience]
        typed_education = [EducationItem(**item) for item in education]

        document = ResumeDocument(
            contact=contact,
            skills=skills,
            summary=self._guess_summary(raw_text),
            experience=typed_experience,
            education=typed_education,
            extras={"fallback": True},
        )
        return document

    def _enrich_document(
        self,
        document: ResumeDocument,
        raw_text: str,
        tokens: Sequence[OCRToken],
    ) -> ResumeDocument:
        contact_update = document.contact.model_dump()
        if not contact_update.get("name"):
            contact_update["name"] = self._guess_name(tokens)
        contact_update = validators.validate_contact_fields(contact_update)
        document.contact = document.contact.model_copy(update=contact_update)

        cleaned_skills = [skill for skill in document.skills if skill and skill.lower() != "not specified"]
        if not cleaned_skills:
            cleaned_skills = self._guess_skills(raw_text)
        document.skills = validators.normalize_skills(cleaned_skills, self.skill_taxonomy)

        if not document.summary:
            document.summary = self._guess_summary(raw_text)

        education, experience = self._extract_structured_entries(raw_text)
        if not document.education and education:
            document.education = education
        if not document.experience and experience:
            document.experience = experience

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
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        capture = False
        for line in lines:
            header = line.strip().rstrip(":").upper()
            if header in {"SKILLS", "TECHNICAL SKILLS"}:
                capture = True
                continue
            if capture and header in {"WORK", "WORK EXPERIENCE", "EXPERIENCE", "EDUCATION"}:
                break
            if capture:
                tokens = [token.strip().title() for token in re_split_skills(line)]
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

    def _extract_structured_entries(self, raw_text: str) -> Tuple[List[Dict], List[Dict]]:
        education: List[Dict] = []
        experience: List[Dict] = []
        lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
        for line in lines:
            if "|" not in line:
                continue
            parts = [segment.strip() for segment in line.split("|")]
            if len(parts) < 3:
                continue
            start, end = self._parse_date_range(parts[-1])
            entry = {
                "institution": parts[0] if "university" in parts[0].lower() or "college" in parts[0].lower() else None,
                "company": parts[0] if "university" not in parts[0].lower() else None,
                "role": parts[1] if "university" not in parts[0].lower() else None,
                "degree": parts[1] if "university" in parts[0].lower() else None,
                "start": start,
                "end": end,
            }

            if entry["institution"]:
                education.append({
                    "institution": entry["institution"],
                    "degree": entry["degree"],
                    "major": None,
                    "start": entry["start"],
                    "end": entry["end"],
                    "details": None,
                })
            elif entry["company"]:
                experience.append({
                    "company": entry["company"],
                    "role": entry["role"],
                    "start": entry["start"],
                    "end": entry["end"],
                    "achievements": [],
                })

        return education, experience

    def _parse_date_range(self, value: str) -> Tuple[Optional[str], Optional[str]]:
        text = value.replace("—", "-").replace("to", "-")
        parts = [segment.strip() for segment in re.split(r"[-–]", text) if segment.strip()]
        if not parts:
            return None, None
        start = self._parse_date(parts[0])
        end = self._parse_date(parts[1]) if len(parts) > 1 else None
        return start, end

    @staticmethod
    def _parse_date(value: str) -> Optional[str]:
        if not value or value.lower() in {"present", "current"}:
            return value.title() if value else None
        parsed = dateparser.parse(value, settings={"PREFER_DAY_OF_MONTH": "first", "DATE_ORDER": "MDY"})
        if parsed:
            return parsed.strftime("%Y-%m")
        return None


def re_split_skills(segment: str) -> List[str]:
    import re

    return [part for part in re.split(r"[•,;|]", segment) if part]
