from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image


@dataclass
class PreprocessResult:
    image: Image.Image
    rotation: float = 0.0
    note: Optional[str] = None


def load_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {path}")
    return image


def to_gray(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def deskew(image: np.ndarray) -> tuple[np.ndarray, float]:
    gray = to_gray(image)
    gray = cv2.bitwise_not(gray)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thresh > 0))
    if coords.size == 0:
        return image, 0.0
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    rotated = cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated, angle


def denoise(image: np.ndarray, strength: int = 10) -> np.ndarray:
    gray = to_gray(image)
    return cv2.fastNlMeansDenoising(gray, h=strength)


def adaptive_threshold(image: np.ndarray) -> np.ndarray:
    gray = to_gray(image)
    return cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        35,
        9,
    )


def enhance_contrast(image: np.ndarray) -> np.ndarray:
    gray = to_gray(image)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def preprocess_for_ocr(path: Path) -> PreprocessResult:
    original = load_image(path)
    deskewed, angle = deskew(original)
    denoised = denoise(deskewed)
    enhanced = enhance_contrast(denoised)
    binary = adaptive_threshold(enhanced)
    pil_image = Image.fromarray(binary)
    return PreprocessResult(image=pil_image, rotation=angle)
