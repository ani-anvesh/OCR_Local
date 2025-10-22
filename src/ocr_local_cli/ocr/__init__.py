"""OCR engine abstractions."""

from .base import BaseOCREngine, OCRToken
from .tesseract_client import TesseractOCREngine

__all__ = ["BaseOCREngine", "OCRToken", "TesseractOCREngine"]
