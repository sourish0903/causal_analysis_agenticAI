# Causal AI Agentic Framework

This repository contains a small Causal AI agentic framework and supporting tools for causal analysis and optimization. It includes utilities for causal forests, helper tools, and an agent-building entrypoint designed to integrate with LLMs and LangGraph.

Please refer scripts/README.md to have more deepdive into the Causal Framework.

Purpose

- Provide reusable causal analysis utilities (causal forests, variable impact, what-if analysis).
- Offer an agent interface that binds LLMs to the causal tools for automated analysis workflows.
- Serve as a local development workspace for experimentation and a Streamlit demo UI.

Quick start — Local development environment

Prerequisites

- macOS with Python 3.13 installed and available as `python3.13`.
- (Optional) `uv` CLI on PATH if you prefer using it to manage virtual environments and installations.

Create the virtual environment

Either (preferred if you have `uv`):

```bash
uv create -p python3.13 causal_agent
```

Or fall back to the built-in venv:

```bash
python3.13 -m venv causal_agent
```

Activate the environment (zsh):

```bash
source causal_agent/bin/activate
```

Install dependencies

Using the provided lockfile (recommended when using `uv`):

```bash
uv pip install --requirements uv.lock
```

Or with pip (fallback):

```bash
pip install --upgrade pip
pip install pandas numpy econml scikit-learn streamlit scipy langchain_openai langchain
```

Verify installation

```bash
python -c "import pandas, numpy, econml, sklearn, streamlit, scipy, langchain, langchain_openai; print('ok')"
```

Running the Streamlit web app

A Streamlit demo app is included at `scripts/streamlit_app.py` in this repository. To run the included demo:

```bash
source causal_agent/bin/activate
streamlit run scripts/streamlit_app.py
```

If you want to run a different Streamlit script, replace the path above with your script's location.

.env and storing secrets

For local development you can store sensitive keys (such as your OpenAI API key) in a local `.env` file based on the provided `.env-example`.

1. Copy the example file to create your local `.env`:

```bash
cp .env-example .env
```

2. Edit `.env` and set your key (replace the placeholder):

```text
OPENAI_API_KEY=sk-...yourkey...
```

3. Ensure `.env` is in your `.gitignore` so you don't accidentally commit secrets. If you don't have a `.gitignore` entry yet, add one:

```bash
echo ".env" >> .gitignore
```

The Streamlit app (`scripts/streamlit_app.py`) will load `.env` using `python-dotenv` if present, so no further action is required after creating the file and activating the virtual environment.

Do NOT commit real API keys to public repositories. Keep secrets local or use secure secret management in CI/CD.

Notes

- The repository provides `pyproject.toml` and `uv.lock` for declarative dependency management. Pin versions in `uv.lock` or `pyproject.toml` if reproducible builds are required.
- Always activate the `causal_agent` environment in any new shell before running code.

If you want, I can add a small example Streamlit app file to `scripts/` and wire a demo using the existing tools.