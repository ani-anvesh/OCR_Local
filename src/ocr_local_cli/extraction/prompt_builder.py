from __future__ import annotations

from pathlib import Path
import re
from collections import defaultdict
from typing import Dict, List, Optional, Sequence

from ..ocr.base import OCRToken


LINE_GAP_PX = 14.0
HEADING_CANDIDATES = {
    "WORK EXPERIENCE": "WORK EXPERIENCE",
    "WORK": "WORK EXPERIENCE",
    "EXPERIENCE": "WORK EXPERIENCE",
    "PROFESSIONAL EXPERIENCE": "WORK EXPERIENCE",
    "EDUCATION": "EDUCATION",
    "SKILLS": "SKILLS",
    "TECHNICAL SKILLS": "SKILLS",
    "PROJECTS": "PROJECTS",
    "CERTIFICATIONS": "CERTIFICATIONS",
    "LANGUAGES": "LANGUAGES",
}

DATE_PATTERN = re.compile(
    r"((Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?\s+\d{4})|\b\d{4}\b",
    re.IGNORECASE,
)


def _center(token: OCRToken) -> tuple[float, float]:
    if token.bbox:
        xs = [float(pt[0]) for pt in token.bbox if isinstance(pt, (list, tuple)) and len(pt) == 2]
        ys = [float(pt[1]) for pt in token.bbox if isinstance(pt, (list, tuple)) and len(pt) == 2]
        if xs and ys:
            return sum(xs) / len(xs), sum(ys) / len(ys)
    return 0.0, float(token.page or 0)


def _token_height(token: OCRToken) -> float:
    if token.bbox:
        ys = [float(pt[1]) for pt in token.bbox if isinstance(pt, (list, tuple)) and len(pt) == 2]
        if ys:
            return max(ys) - min(ys)
    return LINE_GAP_PX


def format_tokens_as_text(tokens: Sequence[OCRToken]) -> str:
    """Aggregate OCR tokens into ordered sections ready for prompting."""

    if not tokens:
        return ""

    grouped: Dict[int, List[OCRToken]] = defaultdict(list)
    for token in tokens:
        grouped[token.page].append(token)

    sections: List[str] = []

    for page in sorted(grouped):
        page_tokens = grouped[page]
        page_tokens.sort(key=lambda t: (_center(t)[1], _center(t)[0]))

        lines: List[List[OCRToken]] = []
        current_line: List[OCRToken] = []
        current_y = None
        current_height = LINE_GAP_PX

        for token in page_tokens:
            if not token.text:
                continue
            _, cy = _center(token)
            height = max(_token_height(token), 6.0)
            gap = current_height * 0.6 + 6.0
            if current_line and current_y is not None and abs(cy - current_y) > gap:
                current_line.sort(key=lambda t: _center(t)[0])
                lines.append(current_line)
                current_line = []
                current_y = cy
                current_height = height
            else:
                current_height = max(current_height, height)
                current_y = cy if current_y is None else (current_y * 0.8 + cy * 0.2)
            current_line.append(token)

        if current_line:
            current_line.sort(key=lambda t: _center(t)[0])
            lines.append(current_line)

        if sections:
            sections.append("")  # blank line between pages

        current_heading = None
        last_structured_entry = False
        for line_tokens in lines:
            line_text = " ".join(_normalise_token_text(tok.text) for tok in line_tokens if tok.text).strip()
            if not line_text:
                continue

            heading = HEADING_CANDIDATES.get(line_text.upper())
            if heading:
                if sections and sections[-1] != "":
                    sections.append("")
                sections.append(f"{heading}:")
                current_heading = heading
                last_structured_entry = False
                continue

            trimmed = _strip_leading_bullet(line_text)

            if current_heading == "SKILLS" and not trimmed.endswith(":"):
                sections.append(_format_skill_line(trimmed))
                continue

            if current_heading == "WORK EXPERIENCE":
                if _looks_like_job_header(trimmed):
                    sections.append(f"- {trimmed}")
                    last_structured_entry = True
                else:
                    prefix = "  - " if last_structured_entry else "- "
                    sections.append(prefix + trimmed)
                continue

            if current_heading == "EDUCATION":
                if _looks_like_education_entry(trimmed):
                    sections.append(f"- {trimmed}")
                    last_structured_entry = True
                else:
                    prefix = "  - " if last_structured_entry else "- "
                    sections.append(prefix + trimmed)
                continue

            if current_heading in {"PROJECTS", "CERTIFICATIONS", "LANGUAGES"}:
                sections.append(f"- {trimmed}")
                last_structured_entry = True
                continue

            sections.append(trimmed)

    # collapse duplicate adjacent lines while maintaining order
    seen = set()
    filtered_sections = []
    for line in sections:
        key = line.strip().lower()
        if line and key in seen:
            continue
        if line:
            seen.add(key)
        filtered_sections.append(line)

    return "\n".join(filtered_sections).strip()


def _strip_leading_bullet(text: str) -> str:
    cleaned = re.sub(r'^\s*[•‣▪●\-\–\—]+\s*', '', text, count=1)
    return cleaned if cleaned else text.strip()


def _contains_date(text: str) -> bool:
    return bool(DATE_PATTERN.search(text))


def _looks_like_job_header(text: str) -> bool:
    if '|' not in text:
        return False
    parts = [segment.strip() for segment in text.split('|') if segment.strip()]
    if len(parts) < 2:
        return False
    return _contains_date(text)


def _looks_like_education_entry(text: str) -> bool:
    if '|' in text and _contains_date(text):
        return True
    upper = text.upper()
    keywords = ("UNIVERSITY", "COLLEGE", "INSTITUTE", "BACHELOR", "MASTER", "GPA")
    return any(word in upper for word in keywords)


def _format_skill_line(text: str) -> str:
    parts = [seg.strip(" -•\t") for seg in re.split(r"[,;\u2022\|]", text) if seg.strip()]
    if not parts:
        return text
    unique = []
    for part in parts:
        if part.lower() not in {p.lower() for p in unique}:
            unique.append(part)
    return "Skills: " + ", ".join(unique)


def _normalise_token_text(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip()
    cleaned = cleaned.replace("–", "-").replace("—", "-")
    cleaned = re.sub(r"(?<=\w)&(?=\w)", " & ", cleaned)
    cleaned = re.sub(r"([a-z])([A-Z])", r"\1 \2", cleaned)
    cleaned = re.sub(r"([.,;:/])(?=\S)", r"\1 ", cleaned)
    cleaned = re.sub(r"(?<=\d)(?=[A-Za-z])", " ", cleaned)
    cleaned = re.sub(r"(?<=[A-Za-z])(?=\d)", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def build_prompt(
    template_path: Path,
    tokens: Sequence[OCRToken],
    layout_notes: Optional[List[str]] = None,
    schema: Optional[Dict] = None,
    raw_text: Optional[str] = None,
) -> str:
    """
    Compose the LLM prompt using a text template and OCR tokens.
    """
    if not template_path.exists():
        raise FileNotFoundError(f"Prompt template missing: {template_path}")

    with template_path.open("r", encoding="utf-8") as handle:
        template = handle.read()

    aggregated_text = raw_text or format_tokens_as_text(tokens)

    layout_section = ""
    if layout_notes:
        layout_section = "Layout notes:\n- " + "\n- ".join(layout_notes)

    schema_section = ""
    if schema:
        schema_section = f"Expected JSON schema:\n{schema}"

    prompt = template.format(
        raw_text=aggregated_text,
        layout_notes=layout_section,
        schema=schema_section,
    )
    return prompt
