from pathlib import Path

from ocr_local_cli.config import load_json_schema
from ocr_local_cli.schema import ResumeDocument


def test_resume_schema_loads(tmp_path: Path):
    schema = load_json_schema(Path("configs/resume_schema.json"))
    assert schema["title"] == "ResumeDocument"

    sample = {
        "contact": {"name": "Jane Doe", "email": "jane@example.com"},
        "skills": ["Python"],
        "education": [{"institution": "State University"}],
        "experience": [{"company": "Acme"}],
    }
    document = ResumeDocument.model_validate(sample)
    assert document.contact.name == "Jane Doe"

