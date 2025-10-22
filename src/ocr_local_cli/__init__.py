"""
High-accuracy resume OCR pipeline package.

The package exposes the `ResumeOCRPipeline` main entry point alongside helper
modules for preprocessing, OCR, layout normalization, LLM-assisted extraction,
and data validation.
"""

from .pipeline import ResumeOCRPipeline

__all__ = ["ResumeOCRPipeline"]

