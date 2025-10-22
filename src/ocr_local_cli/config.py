from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


DEFAULT_CONFIG_PATH = Path("configs/config.yaml")


@dataclass
class OCRConfig:
    """OCR engine configuration options."""

    lang: str = "en"
    engine: str = "paddle"
    enable_angle_cls: bool = True
    cls_batch_num: int = 30
    rec_batch_num: int = 6
    model_root: Optional[str] = None
    cpu_threads: int = 1


@dataclass
class LLMConfig:
    """LLM extraction client configuration."""

    endpoint: str = "http://localhost:8080/extract"
    api_key_env: str = "LLM_API_KEY"
    model: Optional[str] = None
    temperature: float = 0.0
    timeout_seconds: int = 45
    max_retries: int = 2
    request_mode: str = "json-prompt"
    system_prompt: Optional[str] = None
    enforce_json: bool = True

    @property
    def api_key(self) -> Optional[str]:
        """Lookup API key from environment if available."""
        return os.environ.get(self.api_key_env)


@dataclass
class PipelineConfig:
    """Top-level configuration shared across pipeline modules."""

    dpi: int = 300
    min_confidence: float = 0.6
    fallback_confidence: float = 0.3
    tmp_dir: Path = Path("tmp")
    schema_path: Path = Path("configs/resume_schema.json")
    prompt_template_path: Path = Path("configs/prompts/resume_extraction.txt")
    skill_taxonomy_path: Optional[Path] = Path("configs/skills_taxonomy.json")
    enable_layout_analysis: bool = True
    enable_icon_mapping: bool = True

    ocr: OCRConfig = field(default_factory=OCRConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dpi": self.dpi,
            "min_confidence": self.min_confidence,
            "fallback_confidence": self.fallback_confidence,
            "tmp_dir": str(self.tmp_dir),
            "schema_path": str(self.schema_path),
            "prompt_template_path": str(self.prompt_template_path),
            "skill_taxonomy_path": str(self.skill_taxonomy_path)
            if self.skill_taxonomy_path
            else None,
            "enable_layout_analysis": self.enable_layout_analysis,
            "enable_icon_mapping": self.enable_icon_mapping,
            "ocr": self.ocr.__dict__,
            "llm": {
                "endpoint": self.llm.endpoint,
                "api_key_env": self.llm.api_key_env,
                "model": self.llm.model,
                "temperature": self.llm.temperature,
                "timeout_seconds": self.llm.timeout_seconds,
                "max_retries": self.llm.max_retries,
                "request_mode": self.llm.request_mode,
                "system_prompt": self.llm.system_prompt,
                "enforce_json": self.llm.enforce_json,
            },
        }


def load_yaml_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_config(path: Optional[Path] = None) -> PipelineConfig:
    """
    Load pipeline configuration from YAML, falling back to defaults.
    """

    config_path = path or DEFAULT_CONFIG_PATH
    if config_path.exists():
        data = load_yaml_config(config_path)
    else:
        data = {}

    ocr_data: Dict[str, Any] = data.get("ocr", {})
    llm_data: Dict[str, Any] = data.get("llm", {})
    defaults = PipelineConfig()

    config = PipelineConfig(
        dpi=data.get("dpi", defaults.dpi),
        min_confidence=data.get("min_confidence", defaults.min_confidence),
        fallback_confidence=data.get(
            "fallback_confidence", defaults.fallback_confidence
        ),
        tmp_dir=Path(data.get("tmp_dir", defaults.tmp_dir)),
        schema_path=Path(data.get("schema_path", defaults.schema_path)),
        prompt_template_path=Path(
            data.get("prompt_template_path", defaults.prompt_template_path)
        ),
        skill_taxonomy_path=(
            Path(data["skill_taxonomy_path"])
            if data.get("skill_taxonomy_path")
            else defaults.skill_taxonomy_path
        ),
        enable_layout_analysis=data.get(
            "enable_layout_analysis", defaults.enable_layout_analysis
        ),
        enable_icon_mapping=data.get(
            "enable_icon_mapping", defaults.enable_icon_mapping
        ),
        ocr=OCRConfig(
            lang=ocr_data.get("lang", defaults.ocr.lang),
            engine=ocr_data.get("engine", defaults.ocr.engine),
            enable_angle_cls=ocr_data.get(
                "enable_angle_cls", defaults.ocr.enable_angle_cls
            ),
            cls_batch_num=ocr_data.get("cls_batch_num", defaults.ocr.cls_batch_num),
            rec_batch_num=ocr_data.get("rec_batch_num", defaults.ocr.rec_batch_num),
            model_root=ocr_data.get("model_root", defaults.ocr.model_root),
            cpu_threads=ocr_data.get("cpu_threads", defaults.ocr.cpu_threads),
        ),
        llm=LLMConfig(
            endpoint=llm_data.get("endpoint", defaults.llm.endpoint),
            api_key_env=llm_data.get("api_key_env", defaults.llm.api_key_env),
            model=llm_data.get("model", defaults.llm.model),
            temperature=llm_data.get("temperature", defaults.llm.temperature),
            timeout_seconds=llm_data.get(
                "timeout_seconds", defaults.llm.timeout_seconds
            ),
            max_retries=llm_data.get("max_retries", defaults.llm.max_retries),
            request_mode=llm_data.get("request_mode", defaults.llm.request_mode),
            system_prompt=llm_data.get("system_prompt", defaults.llm.system_prompt),
            enforce_json=llm_data.get("enforce_json", defaults.llm.enforce_json),
        ),
    )
    return config


def dump_config(config: PipelineConfig, path: Optional[Path] = None) -> None:
    """Persist the configuration to YAML."""
    config_path = path or DEFAULT_CONFIG_PATH
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config.to_dict(), handle, sort_keys=False)


def load_json_schema(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)
