from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional

import dateparser
from rapidfuzz import fuzz, process

EMAIL_REGEX = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_REGEX = re.compile(r"(\+?\d[\d\s\-().]{7,}\d)")


def extract_email(text: str) -> Optional[str]:
    match = EMAIL_REGEX.search(text)
    return match.group(0) if match else None


def extract_phone(text: str) -> Optional[str]:
    match = PHONE_REGEX.search(text)
    if not match:
        return None
    phone = re.sub(r"[^\d+]", "", match.group(0))
    if len(phone) < 8:
        return None
    return phone


def parse_dates(text: str) -> List[str]:
    tokens = re.findall(r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}", text, flags=re.IGNORECASE)
    normalized: List[str] = []
    for token in tokens:
        parsed = dateparser.parse(token)
        if parsed:
            normalized.append(parsed.strftime("%Y-%m"))
    return normalized


def normalize_skill(skill: str, taxonomy: Optional[List[str]] = None) -> str:
    skill = skill.strip()
    if not taxonomy:
        return skill
    match, score, _ = process.extractOne(skill, taxonomy, scorer=fuzz.token_sort_ratio)
    if score >= 80:
        return match  # type: ignore[return-value]
    return skill


def normalize_skills(skills: Iterable[str], taxonomy: Optional[List[str]] = None) -> List[str]:
    return sorted({normalize_skill(skill, taxonomy) for skill in skills if skill})


def validate_contact_fields(data: Dict[str, str]) -> Dict[str, str]:
    text_blob = " ".join(value for value in data.values() if isinstance(value, str))
    email = extract_email(text_blob)
    phone = extract_phone(text_blob)
    result = dict(data)
    if email:
        result["email"] = email
    if phone:
        result["phone"] = phone
    return result

