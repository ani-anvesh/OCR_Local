from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

from ..ocr.base import OCRToken


def build_prompt(
    template_path: Path,
    tokens: Sequence[OCRToken],
    layout_notes: Optional[List[str]] = None,
    schema: Optional[Dict] = None,
) -> str:
    """
    Compose the LLM prompt using a text template and OCR tokens.
    """
    if not template_path.exists():
        raise FileNotFoundError(f"Prompt template missing: {template_path}")

    with template_path.open("r", encoding="utf-8") as handle:
        template = handle.read()

    sorted_tokens = sorted(
        tokens, key=lambda t: (t.page, t.bbox[0][1] if t.bbox else 0, t.bbox[0][0])
    )
    raw_lines = [token.text for token in sorted_tokens]
    raw_text = "\n".join(raw_lines)

    layout_section = ""
    if layout_notes:
        layout_section = "Layout notes:\n- " + "\n- ".join(layout_notes)

    schema_section = ""
    if schema:
        schema_section = f"Expected JSON schema:\n{schema}"

    prompt = template.format(
        raw_text=raw_text,
        layout_notes=layout_section,
        schema=schema_section,
    )
    return prompt

