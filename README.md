# GitHub Repo Analyzer API

Local FastAPI service that summarizes a GitHub repository into structured JSON using Nebius Token Factory (OpenAI-compatible API).

## Endpoint

- `POST /summarize`
- Request:
  ```json
  {
    "github_url": "https://github.com/psf/requests"
  }
  ```
- Success response:
  ```json
  {
    "summary": "string",
    "technologies": ["string"],
    "structure": ["string"]
  }
  ```
- Error response:
  ```json
  {
    "error": {
      "code": "ERROR_CODE",
      "message": "Human readable explanation",
      "details": {}
    }
  }
  ```

## Architecture

Pipeline:

1. Parse and validate GitHub URL (`owner/repo`).
2. Fetch repo metadata and default branch.
3. Fetch recursive repository tree.
4. Heuristic pre-filter skips generated/binary/asset files.
5. Planner LLM selects important file paths from tree-only input.
6. Fetch only selected file contents from GitHub.
7. Large files are chunked and summarized; smaller files are passed as raw content.
8. Deterministic technology extraction from dependency/config files.
9. Final summarizer LLM produces strict JSON (`summary`, `technologies`, `structure`).
10. Strict output validation and clamping before response.

## Model and Provider

- Provider base URL: `https://api.tokenfactory.nebius.com/v1/`
- Model: `meta-llama/Meta-Llama-3.1-8B-Instruct`
- SDK: official `openai` Python SDK configured with Nebius-compatible base URL.
- Why this model: instruction-tuned and cost-effective for planner + summarization stages.

## Security and Secrets

- API key is read only from environment variable: `NEBIUS_API_KEY`.
- No API key hardcoding in source files.

## Token and Context Controls

- `MAX_SELECTED_FILES = 10`
- `MAX_FILE_BYTES = 200000`
- `MAX_FILE_TOKENS = 8000`
- `CHUNK_TOKENS = 2000`
- `SAFE_CONTEXT = 100000`

Behavior:

- Files above byte limit are skipped.
- Files above token threshold are chunked and summarized.
- Final context is trimmed by dropping low-priority files first.
- If still too large, API returns `TOKEN_OVERFLOW`.

## File Selection and Filtering

Hard skip examples:

- Directories: `node_modules/`, `vendor/`, `third_party/`, `dist/`, `build/`, `out/`, `target/`, `coverage/`
- Caches: `__pycache__/`, `.mypy_cache/`, `.pytest_cache/`
- Artifacts/binaries/assets: image formats, archives, model/data binaries, `*.min.js`, `*.map`

De-prioritized (not hard skipped):

- Lockfiles and test files

Planner fallback:

- Deterministic fallback selection preserves diversity across docs, config, entrypoints, and core source files.

## Technology Extraction

Hybrid approach:

- Deterministic extraction from files like `requirements.txt`, `pyproject.toml`, `package.json`, `go.mod`, `Cargo.toml`, `Dockerfile`, `tsconfig.json`, `setup.py`, `setup.cfg`.
- LLM output is post-filtered to keep technologies grounded in deterministic candidates/languages, with at most two extra evidenced items.

## Reliability Controls

- GitHub timeout: 10s
- LLM timeout: 60s
- Retries:
  - one retry for invalid LLM JSON
  - one retry for LLM timeout
  - exponential backoff for GitHub rate limit responses

## Error Codes

- `INVALID_GITHUB_URL`
- `REPO_NOT_FOUND`
- `GITHUB_RATE_LIMIT`
- `GITHUB_TIMEOUT`
- `GITHUB_API_ERROR`
- `FILE_DECODE_ERROR`
- `LLM_TIMEOUT`
- `LLM_INVALID_RESPONSE`
- `TOKEN_OVERFLOW`
- `INTERNAL_ERROR`

## Run Instructions (Evaluator Flow)

1. Ensure Python is installed.
2. Create virtual environment:
   ```bash
   python -m venv .venv
   ```
3. Activate:
   ```bash
   source .venv/bin/activate
   ```
   Windows:
   ```bat
   .venv\Scripts\activate
   ```
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
5. Set API key:
   ```bash
   export NEBIUS_API_KEY=YOUR_KEY
   ```
   Windows:
   ```bat
   set NEBIUS_API_KEY=YOUR_KEY
   ```
6. Run server:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000
   ```

If your environment only has `python3`, replace `python` with `python3`.

## Curl Test

```bash
curl -X POST http://localhost:8000/summarize \
  -H "Content-Type: application/json" \
  -d '{"github_url": "https://github.com/psf/requests"}'
```

## Environment Variables

- `NEBIUS_API_KEY` (required)
- `NEBIUS_BASE_URL` (optional override, default `https://api.tokenfactory.nebius.com/v1/`)
- `NEBIUS_MODEL` (optional override, default `meta-llama/Meta-Llama-3.1-8B-Instruct`)

## Project Layout

- `main.py`: FastAPI app, endpoint, and error envelope mapping.
- `github_service.py`: GitHub URL parsing, tree/content fetch, filtering, deterministic fallback selection.
- `llm_service.py`: Nebius/OpenAI wrapper with strict JSON parsing + retry policy.
- `summarizer.py`: Orchestration for planner/chunk/final summarize pipeline.
- `models.py`: request/response/error model definitions.
- `config.py`: constants and env-driven settings.
- `utils.py`: token estimation, chunking, and path heuristics.
- `requirements.txt`: runtime dependencies.

## Submission Checklist

- Source files are present in repository root.
- `requirements.txt` includes runtime dependencies.
- `README.md` includes clean-machine setup and curl validation command.
- API key is supplied via `NEBIUS_API_KEY`.
- Ready to archive and upload as zip for submission.
