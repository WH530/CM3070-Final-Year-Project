"""Install and verify the portable Ollama runtime and required local model."""

from __future__ import annotations

import argparse
import signal

try:
    from .local_ollama import (
        LocalRuntimeError,
        install_runtime,
        pull_model,
        smoke_test_model,
        start_server,
        stop_process,
    )
except ImportError:  # Allows `python backend/generation/ollama/setup_local_runtime.py` as well.
    from local_ollama import (  # type: ignore[no-redef]
        LocalRuntimeError,
        install_runtime,
        pull_model,
        smoke_test_model,
        start_server,
        stop_process,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Download the pinned project-local Ollama runtime and Qwen3.5 4B model."
        )
    )
    parser.add_argument(
        "--skip-smoke-test",
        action="store_true",
        help="install the runtime and model without running a short inference test",
    )
    arguments = parser.parse_args()

    def request_shutdown(_signum: int, _frame: object) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, request_shutdown)

    server = None
    try:
        executable = install_runtime()
        server = start_server()
        pull_model(executable)
        if not arguments.skip_smoke_test:
            smoke_test_model()
    except LocalRuntimeError as exc:
        parser.exit(1, f"Local runtime setup failed: {exc}\n")
    except KeyboardInterrupt:
        parser.exit(130, "Local runtime setup cancelled.\n")
    finally:
        stop_process(server)

    print("Portable local-AI setup completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
