#!/usr/bin/env bash
# Always launches api.py with this project's own virtualenv, never whatever
# python happens to be first on PATH (a stray global interpreter is what
# caused the langchain/langchain-core version mismatch bug).
set -euo pipefail
cd "$(dirname "$0")"
./.venv/bin/python -m pip install -q -r requirements.txt
cd backend
exec ../.venv/bin/python api.py
