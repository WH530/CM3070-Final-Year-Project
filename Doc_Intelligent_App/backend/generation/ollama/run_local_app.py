"""Run the project-owned Ollama server and FastAPI as managed child processes."""

from __future__ import annotations

import argparse
import signal
import time
import webbrowser

try:
    from .local_ollama import (
        LocalRuntimeError,
        FASTAPI_BASE,
        OLLAMA_API_BASE,
        OLLAMA_MODEL,
        model_is_installed,
        start_backend,
        start_server,
        stop_process,
        wait_for_backend,
    )
except ImportError:  # Allows `python backend/generation/ollama/run_local_app.py` as well.
    from local_ollama import (  # type: ignore[no-redef]
        LocalRuntimeError,
        FASTAPI_BASE,
        OLLAMA_API_BASE,
        OLLAMA_MODEL,
        model_is_installed,
        start_backend,
        start_server,
        stop_process,
        wait_for_backend,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run portable Ollama and the FastAPI application without reload."
    )
    parser.parse_args()

    def request_shutdown(_signum: int, _frame: object) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, request_shutdown)

    ollama = None
    backend = None
    exit_code = 0
    try:
        ollama = start_server()
        if not model_is_installed():
            raise LocalRuntimeError(
                f"Required model {OLLAMA_MODEL} is not installed. Run "
                "`python -m generation.ollama.setup_local_runtime` first."
            )

        backend = start_backend()
        wait_for_backend(backend)
        print(f"Local LLM: {OLLAMA_API_BASE} ({OLLAMA_MODEL})")
        print(f"Application: {FASTAPI_BASE}")
        print("Press Ctrl+C to stop both managed processes.")
        try:
            browser_opened = webbrowser.open(FASTAPI_BASE)
        except webbrowser.Error:
            browser_opened = False
        if not browser_opened:
            print(f"The browser did not open automatically. Visit {FASTAPI_BASE}.")

        while backend.process.poll() is None:
            if ollama.process.poll() is not None:
                raise LocalRuntimeError(
                    f"Portable Ollama exited unexpectedly with code "
                    f"{ollama.process.returncode}."
                )
            time.sleep(0.5)
        exit_code = backend.process.returncode or 0
    except KeyboardInterrupt:
        print("\nStopping the local application...")
    except LocalRuntimeError as exc:
        parser.exit(1, f"Could not run the local application: {exc}\n")
    finally:
        stop_process(backend)
        stop_process(ollama)

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
