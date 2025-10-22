from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Iterable, List

try:
    from pdf2image import convert_from_path  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    convert_from_path = None  # type: ignore


def pdf_to_images(pdf_path: Path, dpi: int = 300) -> List[Path]:
    """
    Convert a PDF into temporary per-page PNG images.

    Returns a list of image paths stored in a temporary directory that will be
    cleaned up automatically when the application exits.
    """
    if convert_from_path is None:
        raise ImportError(
            "pdf2image is not installed. Install it or convert the PDF to images manually."
        )

    pdf_path = pdf_path.expanduser().resolve()
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    tmp_dir = Path(tempfile.mkdtemp(prefix="resume_ocr_"))
    images = convert_from_path(str(pdf_path), dpi=dpi, fmt="png")
    output_paths: List[Path] = []
    for idx, image in enumerate(images):
        out_path = tmp_dir / f"{pdf_path.stem}_page{idx:03d}.png"
        image.save(out_path, "PNG")
        output_paths.append(out_path)
    return output_paths


def iter_image_paths(path: Path) -> Iterable[Path]:
    """
    Yield all images from a given path. Accepts directories or single images/PDFs.
    """
    path = path.expanduser()
    if path.is_dir():
        for file_path in sorted(path.iterdir()):
            if file_path.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
                yield file_path
    elif path.suffix.lower() == ".pdf":
        for converted in pdf_to_images(path):
            yield converted
    else:
        yield path

