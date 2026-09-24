"""Starts only the project-managed local Ollama server — not the full FastAPI
app — and blocks until Ctrl+C.

api.py's own startup health checks eagerly load BAAI/bge-m3 and
BAAI/bge-reranker-v2-m3 onto the GPU (retrieval/embedder.py,
retrieval/reranker.py check_load()). Running api.py at the same time as
run_eval.py — which loads its own separate copies of those same models —
makes both processes fight over the same GPU, which is why the reranker's
first load took 600+ seconds in testing instead of the usual few seconds.
This script gives the eval sole use of the GPU while still providing the
local Ollama server the local-Qwen provider needs.

    python scripts/run_ollama_only.py   (run from eval/)
"""

from __future__ import annotations

import signal
import sys
import time
from pathlib import Path

# eval/scripts/run_ollama_only.py -> parent (scripts/) -> parent (eval/) -> parent (Doc_Intelligent_App/)
BACKEND_DIR = Path(__file__).resolve().parent.parent.parent / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from generation.ollama.local_ollama import (  # noqa: E402
    LocalRuntimeError,
    OLLAMA_API_BASE,
    OLLAMA_MODEL,
    install_runtime,
    model_is_installed,
    server_is_healthy,
    start_server,
    stop_process,
)


def main() -> int:
    if server_is_healthy():
        print(f"Local Ollama is already running at {OLLAMA_API_BASE}")
        print("(leaving it as-is — nothing for this script to manage)")
        return 0

    def request_shutdown(_signum: int, _frame: object) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, request_shutdown)

    managed = None
    try:
        executable = install_runtime()
        managed = start_server()
        if not model_is_installed():
            print(f"Model {OLLAMA_MODEL} is not installed — run setup_local_runtime.py first.")
            return 1
        print(f"Local Ollama ready at {OLLAMA_API_BASE} ({OLLAMA_MODEL})")
        print("Press Ctrl+C to stop it.")
        while managed.process.poll() is None:
            time.sleep(0.5)
        return managed.process.returncode or 0
    except LocalRuntimeError as exc:
        print(f"Could not start local Ollama: {exc}")
        return 1
    except KeyboardInterrupt:
        print("\nStopping local Ollama...")
        return 0
    finally:
        stop_process(managed)


if __name__ == "__main__":
    raise SystemExit(main())
