#!/usr/bin/env bash
# Always launches api.py with this project's own virtualenv, never whatever
# python happens to be first on PATH (a stray global interpreter is what
# caused the langchain/langchain-core version mismatch bug).
set -euo pipefail
cd "$(dirname "$0")"
# First run on a fresh clone: .venv is git-ignored, so create it.
if [ ! -x .venv/bin/python ]; then
  python3.10 -m venv .venv
fi
./.venv/bin/python -m pip install -q -r requirements.txt
cd backend
exec ../.venv/bin/python api.py
