# OCR Local Resume Parser

End-to-end OCR pipeline optimized for parsing resume PDFs or images and emitting a structured JSON document suitable for recruitment automation.

## Features
- High-quality preprocessing with OpenCV (deskew, denoise, adaptive thresholding).
- Layout-aware OCR using PaddleOCR with multi-column heuristics.
- LLM-assisted entity extraction driven by configurable prompts and schema.
- Rule-based fallback for critical fields plus validation (email, phone, dates, skills).
- Modular architecture for swapping OCR engines, LLM providers, or validators.

## Project Layout
```
configs/                # YAML config, JSON schema, prompt template, skill taxonomy
docs/                   # Architecture overview
samples/                # Place sample resumes here (gitignored)
scripts/run_pipeline.py # CLI entry point
src/ocr_local_cli/      # Core Python package
    preprocessing/      # Image preprocessing utilities
    ocr/                # OCR engine abstractions
    layout/             # Multi-column handling
    extraction/         # Prompt builder + LLM client + response parser
    validation/         # Field validators and normalizers
    utils/              # Logging & file helpers
tests/                  # Basic smoke test for schema
```

## Getting Started
1. **Install system dependencies**
   - Tesseract optional: `brew install tesseract`
   - PaddleOCR prerequisites: `pip install paddlepaddle==2.5.2` (CPU) or GPU variant.
   - Poppler for PDF conversion: `brew install poppler`
2. **Install Python dependencies**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -e .[dev]
   ```
3. **Configure LLM endpoint**
   - Update `configs/config.yaml` with your extraction endpoint.
   - Provide API key in `LLM_API_KEY` (or change the env var in config).

### Using an Open-Source LLM
- Install a local serving stack such as [Ollama](https://ollama.com), [LM Studio](https://lmstudio.ai), or [vLLM](https://github.com/vllm-project/vllm) with OpenAI-compatible APIs.
- Example with Ollama:
  1. `brew install ollama`
  2. `ollama serve` (runs at `http://localhost:11434`)
  3. `ollama pull mistral` (or another resume-friendly model)
- Ensure `configs/config.yaml` points to your local endpoint (default is `http://localhost:11434/v1/chat/completions`) and set `llm.request_mode: chat-completions`.
- The `system_prompt` and prompt template already instruct the model to return strict JSON; leave `enforce_json: false` for providers that do not support OpenAI's JSON schema feature.

## Usage
```bash
python scripts/run_pipeline.py samples/jane_doe_resume.pdf --verbose
```
Results are saved as JSON files under `outputs/`.

To override configuration:
```bash
python scripts/run_pipeline.py data/resumes/*.pdf --config custom_config.yaml
```

## Prompt Engineering
- Edit `configs/prompts/resume_extraction.txt` to adjust instructions.
- JSON schema at `configs/resume_schema.json` ensures consistent responses.
- Few-shot examples can be appended inside the prompt file or injected dynamically.

## Extending
- Implement a new OCR backend by subclassing `BaseOCREngine`.
- Swap LLM providers by providing a custom `LLMClient`.
- Add new validators or enrichment rules inside `src/ocr_local_cli/validation`.

## Tests
```bash
pytest
```

## References
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
- [LayoutParser](https://layout-parser.github.io/)
- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)
