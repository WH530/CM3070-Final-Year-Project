"""Download and manage the project-owned portable Ollama process.

Only Python's standard library is used so the runtime can be installed before
the application's larger machine-learning dependencies are loaded. Downloaded
binaries, model weights, and logs remain under ``.runtime/`` in the project.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import signal
import socket
import stat
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any

APP_ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = APP_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

import config as app_config  # noqa: E402  (path is established above)

RUNTIME_ROOT = APP_ROOT / ".runtime"
OLLAMA_ROOT = RUNTIME_ROOT / "ollama"
MODEL_ROOT = RUNTIME_ROOT / "models" / "ollama"
LOG_ROOT = RUNTIME_ROOT / "logs"
CACHE_ROOT = RUNTIME_ROOT / "cache"

OLLAMA_VERSION = "0.32.13"
OLLAMA_MODEL = app_config.OLLAMA_MODEL
OLLAMA_HOST = app_config.OLLAMA_HOST
OLLAMA_PORT = app_config.OLLAMA_PORT
OLLAMA_API_BASE = app_config.OLLAMA_API_BASE
FASTAPI_HOST = app_config.APP_HOST
FASTAPI_PORT = app_config.APP_PORT
FASTAPI_BASE = app_config.APP_BASE_URL


class LocalRuntimeError(RuntimeError):
    """A portable-runtime setup or lifecycle operation failed safely."""


@dataclass(frozen=True)
class RuntimeArtifact:
    platform_id: str
    url: str
    sha256: str
    size_bytes: int
    archive_type: str
    executable_name: str


# Pinned official release assets. The SHA-256 digests and sizes are published
# by the GitHub Releases API for ollama/ollama v0.32.13.
_MACOS_ARTIFACT = RuntimeArtifact(
    platform_id="macos-universal",
    url=(
        "https://github.com/ollama/ollama/releases/download/v0.32.13/ollama-darwin.tgz"
    ),
    sha256="71efd44f3b5f2019f42bae17ae58eb3de8bd25ce3ca3bc89aea58e53e5d091d1",
    size_bytes=153_973_425,
    archive_type="tar.gz",
    executable_name="ollama",
)

_WINDOWS_X64_ARTIFACT = RuntimeArtifact(
    platform_id="windows-x64",
    url=(
        "https://github.com/ollama/ollama/releases/download/"
        "v0.32.13/ollama-windows-amd64.zip"
    ),
    sha256="20d61a8075038694f5b6db1e937551dbc79d470e85217003facf6ecaac394258",
    size_bytes=1_459_437_863,
    archive_type="zip",
    executable_name="ollama.exe",
)


@dataclass
class ManagedProcess:
    process: subprocess.Popen[Any]
    log_handle: IO[str] | None = None


def select_artifact(
    system_name: str | None = None, machine_name: str | None = None
) -> RuntimeArtifact:
    """Return the pinned artifact for a supported operating system and CPU."""
    system = (system_name or platform.system()).strip().lower()
    machine = (machine_name or platform.machine()).strip().lower()

    if system == "darwin" and machine in {"arm64", "aarch64"}:
        return _MACOS_ARTIFACT
    if system == "windows" and machine in {"x86_64", "amd64"}:
        return _WINDOWS_X64_ARTIFACT

    raise LocalRuntimeError(
        "This application is supported only on macOS with Apple Silicon and "
        "64-bit x86 Windows. Detected "
        f"system={system_name or platform.system()!r}, "
        f"machine={machine_name or platform.machine()!r}."
    )


def install_directory(artifact: RuntimeArtifact | None = None) -> Path:
    selected = artifact or select_artifact()
    return OLLAMA_ROOT / OLLAMA_VERSION / selected.platform_id


def _find_executable(directory: Path, artifact: RuntimeArtifact) -> Path | None:
    candidates = [
        path for path in directory.rglob(artifact.executable_name) if path.is_file()
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda path: len(path.relative_to(directory).parts))


def installed_executable() -> Path:
    """Return the verified installed executable or explain how to install it."""
    artifact = select_artifact()
    directory = install_directory(artifact)
    marker_path = directory / ".install.json"
    executable = _find_executable(directory, artifact) if directory.exists() else None

    if not directory.exists():
        raise LocalRuntimeError(
            "The project-local Ollama runtime is not installed. Run "
            "`python -m generation.ollama.setup_local_runtime` from backend/ first."
        )
    if not marker_path.is_file() or executable is None:
        raise LocalRuntimeError(
            "The project-local Ollama installation is incomplete. Remove "
            f"{directory} and run setup again."
        )

    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LocalRuntimeError(
            f"The portable runtime marker is unreadable: {marker_path}"
        ) from exc

    expected = {
        "version": OLLAMA_VERSION,
        "platform_id": artifact.platform_id,
        "artifact_sha256": artifact.sha256,
    }
    if any(marker.get(key) != value for key, value in expected.items()):
        raise LocalRuntimeError(
            "The existing portable runtime was not installed from the pinned "
            f"artifact. Remove {directory} and run setup again."
        )

    return executable


def _download(artifact: RuntimeArtifact, destination: Path) -> None:
    request = urllib.request.Request(
        artifact.url,
        headers={"User-Agent": "CM3070-Document-Intelligence-Setup/1.0"},
    )
    digest = hashlib.sha256()
    downloaded = 0
    next_report = 10

    print(f"Downloading Ollama {OLLAMA_VERSION} for {artifact.platform_id}...")
    try:
        with (
            urllib.request.urlopen(request, timeout=60) as response,
            destination.open("wb") as output,
        ):
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
                digest.update(chunk)
                downloaded += len(chunk)
                percent = downloaded * 100 // artifact.size_bytes
                if percent >= next_report:
                    print(f"  {min(percent, 100)}%")
                    next_report += 10
    except (OSError, urllib.error.URLError) as exc:
        raise LocalRuntimeError(f"Could not download {artifact.url}: {exc}") from exc

    if downloaded != artifact.size_bytes:
        raise LocalRuntimeError(
            "Downloaded runtime size does not match the pinned release: "
            f"expected {artifact.size_bytes} bytes, received {downloaded} bytes."
        )
    actual_digest = digest.hexdigest()
    if actual_digest != artifact.sha256:
        raise LocalRuntimeError(
            "Downloaded runtime failed SHA-256 verification: "
            f"expected {artifact.sha256}, received {actual_digest}."
        )


def _validated_destination(root: Path, member_name: str) -> Path:
    destination = (root / member_name).resolve()
    _ensure_inside(root, destination, member_name)
    return destination


def _ensure_inside(root: Path, destination: Path, member_name: str) -> None:
    try:
        destination.relative_to(root.resolve())
    except ValueError as exc:
        raise LocalRuntimeError(
            f"The runtime archive contains an unsafe path: {member_name!r}."
        ) from exc


def _extract_archive(archive_path: Path, destination: Path, archive_type: str) -> None:
    destination.mkdir(parents=True, exist_ok=False)

    try:
        if archive_type == "zip":
            with zipfile.ZipFile(archive_path) as archive:
                for member in archive.infolist():
                    _validated_destination(destination, member.filename)
                archive.extractall(destination)
            return

        if archive_type == "tar.gz":
            with tarfile.open(archive_path, mode="r:gz") as archive:
                for member in archive.getmembers():
                    member_destination = _validated_destination(
                        destination, member.name
                    )
                    if member.issym():
                        link_destination = (
                            member_destination.parent / member.linkname
                        ).resolve()
                        _ensure_inside(destination, link_destination, member.linkname)
                    elif member.islnk():
                        link_destination = (destination / member.linkname).resolve()
                        _ensure_inside(destination, link_destination, member.linkname)

                # Python 3.10.12+ provides the hardened data filter. Retain
                # the explicit checks above for older compatible releases.
                if hasattr(tarfile, "data_filter"):
                    archive.extractall(destination, filter="data")
                else:
                    archive.extractall(destination)
            return
    except LocalRuntimeError:
        raise
    except (OSError, tarfile.TarError, zipfile.BadZipFile) as exc:
        raise LocalRuntimeError(
            f"Could not extract the verified runtime: {exc}"
        ) from exc

    raise LocalRuntimeError(f"Unsupported runtime archive type: {archive_type}")


def install_runtime() -> Path:
    """Download, verify, and install the pinned runtime into ``.runtime``."""
    artifact = select_artifact()
    directory = install_directory(artifact)

    if directory.exists():
        return installed_executable()

    directory.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="ollama-install-", dir=directory.parent
    ) as temporary_name:
        temporary = Path(temporary_name)
        archive_path = temporary / "runtime.archive"
        extracted = temporary / "extracted"
        _download(artifact, archive_path)
        print("Verifying and extracting the portable runtime...")
        _extract_archive(archive_path, extracted, artifact.archive_type)

        executable = _find_executable(extracted, artifact)
        if executable is None:
            raise LocalRuntimeError(
                f"The verified archive did not contain {artifact.executable_name}."
            )
        if os.name != "nt":
            executable.chmod(executable.stat().st_mode | stat.S_IXUSR)

        marker = {
            "version": OLLAMA_VERSION,
            "platform_id": artifact.platform_id,
            "artifact_url": artifact.url,
            "artifact_sha256": artifact.sha256,
        }
        try:
            (extracted / ".install.json").write_text(
                json.dumps(marker, indent=2) + "\n", encoding="utf-8"
            )
            os.replace(extracted, directory)
        except OSError as exc:
            raise LocalRuntimeError(
                f"Could not install the verified runtime at {directory}: {exc}"
            ) from exc

    executable = installed_executable()
    print(f"Portable Ollama installed at {executable}")
    return executable


def ollama_environment() -> dict[str, str]:
    """Build the environment inherited by the project-owned server and CLI."""
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update(
        {
            "OLLAMA_MODELS": str(MODEL_ROOT),
            "OLLAMA_HOST": f"{OLLAMA_HOST}:{OLLAMA_PORT}",
            "OLLAMA_NO_CLOUD": "1",
            "OLLAMA_CONTEXT_LENGTH": str(app_config.OLLAMA_CONTEXT_LENGTH),
            "OLLAMA_NUM_PARALLEL": str(app_config.OLLAMA_NUM_PARALLEL),
            "OLLAMA_MAX_LOADED_MODELS": str(app_config.OLLAMA_MAX_LOADED_MODELS),
            "OLLAMA_KEEP_ALIVE": app_config.OLLAMA_KEEP_ALIVE,
        }
    )
    return environment


def _request_json(
    path: str, payload: dict[str, Any] | None = None, timeout: float = 2
) -> dict[str, Any]:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{OLLAMA_API_BASE}{path}",
        data=body,
        headers={"Content-Type": "application/json"},
        method="GET" if body is None else "POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        parsed = json.loads(response.read().decode("utf-8"))
    if not isinstance(parsed, dict):
        raise ValueError("Ollama returned a non-object JSON response.")
    return parsed


def server_is_healthy() -> bool:
    try:
        response = _request_json("/api/version")
    except (OSError, ValueError, urllib.error.URLError):
        return False
    return isinstance(response.get("version"), str)


def _port_is_in_use(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.5):
            return True
    except OSError:
        return False


def _subprocess_group_options() -> dict[str, Any]:
    if os.name == "nt":
        return {"creationflags": getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)}
    return {"start_new_session": True}


def _tail_log(path: Path, line_count: int = 20) -> str:
    try:
        return "\n".join(path.read_text(encoding="utf-8").splitlines()[-line_count:])
    except OSError:
        return "(Ollama log could not be read.)"


def start_server(timeout: float = 60) -> ManagedProcess:
    """Start only the project-owned server and wait for its health endpoint."""
    if _port_is_in_use(OLLAMA_HOST, OLLAMA_PORT):
        raise LocalRuntimeError(
            f"Port {OLLAMA_PORT} is already in use. The project will not reuse or "
            "terminate another process; close that process and try again."
        )

    executable = installed_executable()
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    log_path = LOG_ROOT / "ollama.log"
    log_handle = log_path.open("a", encoding="utf-8", buffering=1)
    log_handle.write(f"\n--- starting portable Ollama {OLLAMA_VERSION} ---\n")

    try:
        process = subprocess.Popen(
            [str(executable), "serve"],
            cwd=executable.parent,
            env=ollama_environment(),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            **_subprocess_group_options(),
        )
    except OSError as exc:
        log_handle.close()
        raise LocalRuntimeError(f"Could not start portable Ollama: {exc}") from exc

    managed = ManagedProcess(process=process, log_handle=log_handle)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            log_handle.flush()
            details = _tail_log(log_path)
            log_handle.close()
            raise LocalRuntimeError(
                f"Portable Ollama exited during startup (code {process.returncode}).\n"
                f"Last log lines:\n{details}"
            )
        if server_is_healthy():
            print(f"Portable Ollama is ready at {OLLAMA_API_BASE}")
            return managed
        time.sleep(0.25)

    stop_process(managed)
    raise LocalRuntimeError(
        f"Portable Ollama did not become healthy within {timeout:.0f} seconds. "
        f"See {log_path}."
    )


def stop_process(managed: ManagedProcess | None, timeout: float = 10) -> None:
    """Stop exactly the child process represented by ``managed``."""
    if managed is None:
        return
    process = managed.process
    if process.poll() is None:
        try:
            if os.name == "nt":
                ctrl_break = getattr(signal, "CTRL_BREAK_EVENT", None)
                if ctrl_break is None:
                    process.terminate()
                else:
                    process.send_signal(ctrl_break)
            else:
                os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=timeout)
        except (OSError, ValueError, subprocess.TimeoutExpired):
            if process.poll() is None:
                try:
                    if os.name == "nt":
                        # `/T` is limited to the exact child PID and its process
                        # tree, preventing an Ollama runner from being orphaned.
                        subprocess.run(
                            [
                                "taskkill",
                                "/PID",
                                str(process.pid),
                                "/T",
                                "/F",
                            ],
                            check=False,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                        )
                    else:
                        os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=5)
                except (OSError, subprocess.TimeoutExpired):
                    if process.poll() is None:
                        try:
                            process.kill()
                        except OSError:
                            pass
    if managed.log_handle is not None and not managed.log_handle.closed:
        managed.log_handle.close()


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _find_listening_pid(port: int) -> int | None:
    """Best-effort lookup of the PID bound to a local TCP port."""
    try:
        if os.name == "nt":
            result = subprocess.run(
                ["netstat", "-ano"], capture_output=True, text=True, check=False
            )
            for line in result.stdout.splitlines():
                if f":{port} " in line and "LISTENING" in line:
                    return int(line.split()[-1])
            return None
        result = subprocess.run(
            ["lsof", "-t", f"-i:{port}", "-sTCP:LISTEN"],
            capture_output=True,
            text=True,
            check=False,
        )
        pids = [int(p) for p in result.stdout.split() if p.strip().isdigit()]
        return pids[0] if pids else None
    except (OSError, ValueError):
        return None


def stop_server(timeout: float = 10) -> None:
    """Stop whatever is listening on OLLAMA_PORT, even if this process
    didn't start it itself (e.g. left running by an earlier ``python
    api.py``) — so stopping the app reliably stops Ollama with it."""
    pid = _find_listening_pid(OLLAMA_PORT)
    if pid is None:
        return
    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(pid), "/T", "/F"],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return
        os.killpg(pid, signal.SIGTERM)
    except OSError:
        return

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and _pid_alive(pid):
        time.sleep(0.1)
    if _pid_alive(pid):
        try:
            os.killpg(pid, signal.SIGKILL)
        except OSError:
            pass


def model_is_installed() -> bool:
    try:
        response = _request_json("/api/tags", timeout=5)
    except (OSError, ValueError, urllib.error.URLError):
        return False

    expected = OLLAMA_MODEL.casefold()
    for model in response.get("models", []):
        if not isinstance(model, dict):
            continue
        names = {str(model.get("name", "")), str(model.get("model", ""))}
        if expected in {name.casefold() for name in names}:
            return True
    return False


def pull_model(executable: Path) -> None:
    """Pull the exact pinned model using the project-owned running server."""
    if model_is_installed():
        print(f"Model {OLLAMA_MODEL} is already installed in {MODEL_ROOT}")
        return

    print(f"Downloading local model {OLLAMA_MODEL} (approximately 3.4 GB)...")
    try:
        result = subprocess.run(
            [str(executable), "pull", OLLAMA_MODEL],
            cwd=executable.parent,
            env=ollama_environment(),
            check=False,
        )
    except OSError as exc:
        raise LocalRuntimeError(f"Could not run Ollama model pull: {exc}") from exc
    if result.returncode != 0 or not model_is_installed():
        raise LocalRuntimeError(
            f"Ollama could not install the required model {OLLAMA_MODEL}."
        )
    print(f"Model {OLLAMA_MODEL} is ready in {MODEL_ROOT}")


def smoke_test_model(timeout: float = 600) -> None:
    """Load the local model and verify that an answer can be generated."""
    print("Running a short local inference test...")
    try:
        response = _request_json(
            "/api/generate",
            payload={
                "model": OLLAMA_MODEL,
                "prompt": "Reply with OK.",
                "stream": False,
                "think": False,
                "options": {"num_predict": 16, "temperature": 0},
                "keep_alive": 0,
            },
            timeout=timeout,
        )
    except (OSError, ValueError, urllib.error.URLError) as exc:
        raise LocalRuntimeError(f"Local model inference test failed: {exc}") from exc
    if (
        response.get("done") is not True
        or not str(response.get("response", "")).strip()
    ):
        raise LocalRuntimeError(
            "Local model inference test returned an invalid response."
        )
    print("Local model inference test passed.")


def start_backend() -> ManagedProcess:
    """Start FastAPI with the active Python interpreter and no reload process."""
    if _port_is_in_use(FASTAPI_HOST, FASTAPI_PORT):
        raise LocalRuntimeError(
            f"Port {FASTAPI_PORT} is already in use. Stop the existing web "
            "application before starting this managed instance."
        )

    backend_directory = APP_ROOT / "backend"
    managed_huggingface_cache = CACHE_ROOT / "huggingface"
    managed_huggingface_cache.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    environment["HF_HOME"] = str(managed_huggingface_cache)
    try:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "api:app",
                "--host",
                FASTAPI_HOST,
                "--port",
                str(FASTAPI_PORT),
            ],
            cwd=backend_directory,
            env=environment,
            **_subprocess_group_options(),
        )
    except OSError as exc:
        raise LocalRuntimeError(f"Could not start FastAPI: {exc}") from exc
    return ManagedProcess(process=process)


def backend_health(timeout: float = 2) -> dict[str, Any] | None:
    """Return FastAPI's health payload, or ``None`` while it is unavailable."""
    request = urllib.request.Request(f"{FASTAPI_BASE}/api/health", method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            parsed = json.loads(response.read().decode("utf-8"))
    except (OSError, ValueError, urllib.error.URLError):
        return None
    return parsed if isinstance(parsed, dict) else None


def wait_for_backend(managed: ManagedProcess, timeout: float = 600) -> None:
    """Wait for FastAPI startup checks and its ``/api/health`` response."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if managed.process.poll() is not None:
            raise LocalRuntimeError(
                "FastAPI exited before becoming healthy "
                f"(code {managed.process.returncode})."
            )
        health = backend_health()
        if health is not None and health.get("status") == "ok":
            print(f"FastAPI is ready at {FASTAPI_BASE}")
            return
        if health is not None and health.get("status") == "error":
            failed_components = [
                f"{name}: {item.get('detail', 'unavailable')}"
                for name, item in health.get("components", {}).items()
                if isinstance(item, dict) and not item.get("available", False)
            ]
            providers = health.get("providers", {})
            provider_ready = any(
                isinstance(item, dict) and item.get("available", False)
                for item in providers.values()
            )
            if not provider_ready:
                failed_components.append("no generation provider is ready")
            details = "; ".join(failed_components) or "startup checks failed"
            raise LocalRuntimeError(f"FastAPI started but is not ready: {details}.")
        time.sleep(0.25)
    raise LocalRuntimeError(
        f"FastAPI did not become healthy at {FASTAPI_BASE}/api/health "
        f"within {timeout:.0f} seconds."
    )
