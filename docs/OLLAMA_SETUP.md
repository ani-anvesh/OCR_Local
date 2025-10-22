# Running the Resume Parser with Ollama

Follow these steps to use an open-source model served locally via [Ollama](https://ollama.com/) on macOS (Apple Silicon or Intel).

## 1. Install and Start Ollama
```bash
brew install ollama
ollama serve
```
`ollama serve` launches the daemon on `http://localhost:11434`. Leave it running in its own terminal window.

## 2. Pull a Suitable Model
Grab a model that handles JSON instructions reliably. `mistral` works well as a general-purpose model:
```bash
ollama pull mistral
```
You can experiment with other models (e.g., `llama3:8b`, `qwen2`) to see which balances quality and speed on your hardware.

## 3. Point the Pipeline at Ollama
The repository ships with `configs/config.ollama.yaml`, already configured for Ollama:
```bash
python scripts/run_pipeline.py samples/jane_doe_resume.pdf \
  --config configs/config.ollama.yaml \
  --verbose
```
Results will be written to the `outputs/` directory.

## 4. Environment Variables (Optional)
- The default configuration does **not** require an API key.
- If you use sandboxed scripts that expect an API key, set `LLM_API_KEY` to any placeholder value (e.g., `export LLM_API_KEY=dummy`).

## 5. Troubleshooting
- **Empty or malformed JSON**: models without JSON mode may return text. The pipeline attempts to parse raw strings; if parsing fails, we fall back to heuristics and mark `used_fallback: true` in the output.
- **Slow responses**: try a smaller model (e.g., `mistral:7b-instruct`) or reduce `prompt_template` size.
- **Model hallucinations**: reinforce strictness by adding examples to `configs/prompts/resume_extraction.txt` or tweak the `system_prompt` in the config.

## 6. Advanced
- To enable response-format enforcement with providers that support OpenAI JSON schema, set `llm.enforce_json: true`.
- For batch processing, wrap the CLI in shell scripts or integrate the pipeline class directly in your application.

