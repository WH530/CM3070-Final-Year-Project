# Always launches api.py with this project's own virtualenv, never whatever
# python happens to be first on PATH (a stray global interpreter is what
# caused the langchain/langchain-core version mismatch bug).
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
# First run on a fresh clone: .venv is git-ignored, so create it.
if (-not (Test-Path .\.venv\Scripts\python.exe)) {
    py -3.10 -m venv .venv
}
& .\.venv\Scripts\python.exe -m pip install -q -r requirements.txt
Set-Location backend
& ..\.venv\Scripts\python.exe api.py
