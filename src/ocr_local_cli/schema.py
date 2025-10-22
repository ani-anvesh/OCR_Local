from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field, HttpUrl, constr


class ContactInfo(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    location: Optional[str] = None
    linkedin: Optional[HttpUrl] = None
    website: Optional[HttpUrl] = None


class EducationItem(BaseModel):
    institution: str
    degree: Optional[str] = None
    major: Optional[str] = None
    start: Optional[str] = Field(None, description="YYYY-MM or YYYY")
    end: Optional[str] = Field(None, description="YYYY-MM or YYYY")
    details: Optional[str] = None


class ExperienceItem(BaseModel):
    company: str
    role: Optional[str] = None
    start: Optional[str] = Field(None, description="YYYY-MM or YYYY")
    end: Optional[str] = Field(None, description="YYYY-MM or YYYY or 'Present'")
    achievements: List[str] = Field(default_factory=list)


class ProjectItem(BaseModel):
    name: str
    description: Optional[str] = None
    technologies: List[str] = Field(default_factory=list)


class ResumeDocument(BaseModel):
    contact: ContactInfo = Field(default_factory=ContactInfo)
    summary: Optional[str] = None
    skills: List[str] = Field(default_factory=list)
    education: List[EducationItem] = Field(default_factory=list)
    experience: List[ExperienceItem] = Field(default_factory=list)
    projects: List[ProjectItem] = Field(default_factory=list)
    certifications: List[str] = Field(default_factory=list)
    languages: List[str] = Field(default_factory=list)
    extras: dict = Field(default_factory=dict)

    def ordered(self) -> dict:
        """Return JSON-compatible ordered dict."""
        return self.model_dump(mode="json", exclude_none=True)


def load_schema(path: Path) -> dict:
    import json

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)

