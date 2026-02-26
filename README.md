# GitHub Repo Analyzer API

FastAPI service that summarizes a public GitHub repository into strict JSON via `POST /summarize`.

## 1) Setup (Clean Machine)

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Windows (PowerShell):

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 2) Environment Variables

Required:

```bash
export NEBIUS_API_KEY="YOUR_NEBIUS_API_KEY"
```

Optional (recommended for GitHub API rate limits):

```bash
export GITHUB_TOKEN="YOUR_GITHUB_PAT"
```

Optional evaluation-only flags (not needed for normal `/summarize` usage):

```bash
export ENVIRONMENT="eval"
export ENABLE_JUDGE="1"
export EVAL_MODEL="meta-llama/Llama-3.3-70B-Instruct-fast"
export EVAL_SUMMARIZE_TIMEOUT_SECONDS="300"
```

Do not commit real keys to source control.

## 3) Run Server (Single Entrypoint)

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## 4) Test With curl

```bash
curl -X POST http://localhost:8000/summarize \
  -H "Content-Type: application/json" \
  -d '{"github_url": "https://github.com/psf/requests"}'
```

## 5) Model Choice

Baseline summarizer model is `meta-llama/Meta-Llama-3.1-8B-Instruct`.
It is instruction-following, cost-efficient, and suitable for long-context summarization workloads (128k context-class usage profile).

Optional override for difficult repositories:

```bash
export SUMMARIZER_MODEL="meta-llama/Llama-3.3-70B-Instruct-fast"
```

## 6) Repository Handling Approach

The pipeline:

1. Fetches repository metadata and recursive file tree.
2. Applies heuristic pre-filtering (skip binaries/assets, `node_modules/`, `vendor/`, caches, etc.).
3. Uses a planner step to select high-signal files under a file budget.
4. Chunks very large files and summarizes chunk outputs before final synthesis.
5. Runs deterministic evidence validation/grounding for technologies and structure claims.

## Optional Evaluation Mode

`python scripts/eval_repo.py <github_url>` calls local `/summarize` and then runs optional LLM judging.

Enabled only when:

- `ENVIRONMENT=eval`
- `ENABLE_JUDGE=1`

Exit codes:

- `0`: pass (`overall >= 0.75` and no hallucination flags)
- `1`: evaluated but failed threshold/flags or runtime error
- `2`: evaluation mode disabled

## Create Submission Zip

Use the included cross-platform script from repo root:

```bash
python scripts/make_zip.py
```

This creates `submission.zip` and excludes:

- `.git/`, `venv/`, `.venv/`, `__pycache__/`, `.pytest_cache/`
- `.env`, `.DS_Store`, editor folders
- existing `submission.zip`

## Grader Sanity Commands

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
export NEBIUS_API_KEY="..."
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Then run the curl test above.
