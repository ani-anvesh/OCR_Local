# OCR Resume Parser Architecture

## High-Level Flow
1. **Ingestion**
   - Accept PDF or image resumes.
   - Convert multi-page PDFs into page images using `pdf2image`.
2. **Preprocessing**
   - Deskew, denoise, and binarize with OpenCV.
   - Optional super-resolution for low-quality scans (Real-ESRGAN hook).
3. **Layout Analysis**
   - Detect columns and reading order via connected components.
   - Capture region metadata for tables, headers, and icon clusters.
4. **OCR**
   - Default to `PaddleOCR` for detection + recognition.
   - Optional fallback to `pytesseract` to improve recall.
5. **Entity Extraction**
   - Build structured prompt containing OCR text + layout hints.
   - Send to LLM (Codex/OpenAI compatible client) for JSON extraction.
   - Merge with rule-based heuristics for critical fields.
6. **Validation & Enrichment**
   - Validate email/phone/date formats.
   - Normalize skills using controlled vocabularies and fuzzy matching.
   - Flag low-confidence extractions for review.
7. **Output**
   - Emit JSON adhering to `configs/resume_schema.json`.
   - Persist results or push to downstream queue (outside scope of this repo).

## Components
- `src/ocr_local_cli/preprocessing/image_ops.py`
  - Image loading, deskew, denoise, thresholding utilities.
- `src/ocr_local_cli/ocr/paddle_client.py`
  - Wrapper around `PaddleOCR` with confidence handling and layout packaging.
- `src/ocr_local_cli/layout/column_merger.py`
  - Layout heuristics to reorder tokens in multi-column resumes.
- `src/ocr_local_cli/extraction/prompt_builder.py`
  - Template-driven prompt builder for LLM extraction.
- `src/ocr_local_cli/extraction/llm_client.py`
  - Synchronous client for local/remote LLM endpoints (OpenAI-compatible, Ollama, etc.) with retry logic.
- `src/ocr_local_cli/extraction/parser.py`
  - Combines OCR tokens, layout data, and LLM response into structured output.
- `src/ocr_local_cli/validation/validators.py`
  - Field-level validation, cleaning, and enrichment.
- `src/ocr_local_cli/pipeline.py`
  - Orchestrates the end-to-end processing pipeline.
- `scripts/run_pipeline.py`
  - CLI entry point for batch or single-file runs.

## Cross-Cutting Concerns
- **Configuration** via `configs/config.yaml`, environment variables, or CLI overrides.
- **Logging** configured centrally in `src/ocr_local_cli/utils/logging.py`.
- **Extensibility**
  - Replace OCR backend by implementing `BaseOCREngine`.
  - Swap LLM providers by extending `LLMClient`.
  - Inject custom validators via plugin registry.
- **Error Handling**
  - Collect per-stage metrics and raise `PipelineError` with context.
  - Provide manual review hooks for low-confidence fields.
