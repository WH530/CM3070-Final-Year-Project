# Always launches api.py with this project's own virtualenv, never whatever
# python happens to be first on PATH (a stray global interpreter is what
# caused the langchain/langchain-core version mismatch bug).
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
& .\.venv\Scripts\python.exe -m pip install -q -r requirements.txt
Set-Location backend
& ..\.venv\Scripts\python.exe api.py
