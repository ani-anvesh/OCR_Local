from __future__ import annotations

import re
from pathlib import Path
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Sequence

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

CONTACT_PATTERN = re.compile(
    r"(@|https?://|www\.|linkedin\.com|github\.com|\+\d|\d{3}[\s)-]\d{3})",
    re.IGNORECASE,
)
URL_SPLIT_PATTERN = re.compile(r"(https?://\S+)")
MULTISPACE_PATTERN = re.compile(r"\s{2,}")


@dataclass
class SectionBuffer:
    name: str
    lines: List[str] = field(default_factory=list)
    state: Dict[str, bool] = field(default_factory=dict)

    def add(self, content: Iterable[str]) -> None:
        for line in content:
            stripped = line.strip()
            if stripped:
                self.lines.append(line.rstrip())


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

    ordered_tokens: List[List[OCRToken]] = []
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

        ordered_tokens.extend(lines)

    section_lookup: Dict[str, SectionBuffer] = {}
    section_order: List[str] = []

    def ensure_section(name: str) -> SectionBuffer:
        if name not in section_lookup:
            section_lookup[name] = SectionBuffer(name=name)
            section_order.append(name)
        return section_lookup[name]

    current_heading: Optional[str] = None

    for line_tokens in ordered_tokens:
        line_text = " ".join(_normalise_token_text(tok.text) for tok in line_tokens if tok.text).strip()
        if not line_text:
            continue

        heading = HEADING_CANDIDATES.get(line_text.upper())
        if heading:
            current_heading = heading
            ensure_section(heading)
            continue

        trimmed = _strip_leading_bullet(line_text)
        section_name = current_heading or _infer_default_section(trimmed)
        buffer = ensure_section(section_name)

        formatted_lines = _format_section_line(section_name, trimmed, buffer.state)
        buffer.add(formatted_lines)

    output_lines: List[str] = []
    for name in section_order:
        buffer = section_lookup[name]
        if not buffer.lines:
            continue
        output_lines.append(f"[SECTION: {name}]")
        if name == "SUMMARY":
            paragraph = " ".join(buffer.lines)
            paragraph = MULTISPACE_PATTERN.sub(" ", paragraph).strip()
            if paragraph:
                output_lines.append(paragraph)
        else:
            seen: set[str] = set()
            for line in buffer.lines:
                key = line.lower()
                if key in seen:
                    continue
                seen.add(key)
                output_lines.append(line)
        output_lines.append("")

    return "\n".join(output_lines).strip()


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
    return "- " + ", ".join(unique)


def _normalise_token_text(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip()
    cleaned = cleaned.replace("–", "-").replace("—", "-")
    cleaned = re.sub(r"(?<=\w)&(?=\w)", " & ", cleaned)
    cleaned = re.sub(r"([a-z])([A-Z])", r"\1 \2", cleaned)
    cleaned = re.sub(r"([.,;:/])(?=\S)", r"\1 ", cleaned)
    cleaned = re.sub(r"(?<=\d)(?=[A-Za-z])", " ", cleaned)
    cleaned = re.sub(r"(?<=[A-Za-z])(?=\d)", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned = re.sub(r"(?<=\w)(https?://)", r" \1", cleaned)
    cleaned = cleaned.replace("Cl/c", "CI/CD")
    cleaned = cleaned.replace("FAIsSS", "FAISS")
    cleaned = cleaned.replace("Graana", "Grafana")
    cleaned = cleaned.replace("rometheus", "Prometheus")
    return cleaned


def _infer_default_section(text: str) -> str:
    if _looks_like_contact_line(text):
        return "CONTACT"
    if _looks_like_name_line(text):
        return "HEADER"
    return "SUMMARY"


def _looks_like_contact_line(text: str) -> bool:
    if "|" in text:
        return True
    if CONTACT_PATTERN.search(text):
        return True
    return False


def _looks_like_name_line(text: str) -> bool:
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return False
    upper = sum(1 for ch in letters if ch.isupper())
    return upper / len(letters) >= 0.6 and 1 <= len(text.split()) <= 5


def _format_section_line(section: str, text: str, state: Dict[str, bool]) -> List[str]:
    if section == "CONTACT":
        return _format_contact_line(text)
    if section == "SKILLS":
        return [_format_skill_line(text)]
    if section == "WORK EXPERIENCE":
        if _looks_like_job_header(text):
            state["role_open"] = True
            return [f"[ROLE] {text}"]
        prefix = "  - " if state.get("role_open") else "- "
        return [prefix + text]
    if section == "EDUCATION":
        if _looks_like_education_entry(text):
            state["edu_open"] = True
            return [f"[EDU] {text}"]
        prefix = "  - " if state.get("edu_open") else "- "
        return [prefix + text]
    if section in {"PROJECTS", "CERTIFICATIONS", "LANGUAGES"}:
        return ["- " + text]
    return [text]


def _format_contact_line(text: str) -> List[str]:
    segments: List[str] = []
    for part in URL_SPLIT_PATTERN.split(text):
        if not part:
            continue
        if URL_SPLIT_PATTERN.fullmatch(part):
            segments.append(part.strip())
        else:
            for piece in part.split("|"):
                clean_piece = MULTISPACE_PATTERN.sub(" ", piece).strip(" -•\t")
                if not clean_piece:
                    continue
                fragments = [frag.strip() for frag in re.split(r"[•\u2022]", clean_piece) if frag.strip()]
                segments.extend(fragments or [clean_piece])
    formatted: List[str] = []
    for segment in segments:
        cleaned = MULTISPACE_PATTERN.sub(" ", segment).strip(" -•\t")
        if cleaned:
            formatted.append(f"- {cleaned}")
    return formatted or [text]


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
