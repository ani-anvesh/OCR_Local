"""OCR engine abstractions."""

from .base import BaseOCREngine, OCRToken
from .paddle_client import PaddleOCREngine

__all__ = ["BaseOCREngine", "OCRToken", "PaddleOCREngine"]

