"""Terminal trace logging for long-running DocAI pipelines."""

from __future__ import annotations

import logging
import sys
import time
from contextlib import contextmanager
from typing import Any, Callable

if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

TRACE_LOGGER = logging.getLogger("docai.trace")
if not TRACE_LOGGER.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(message)s"))
    TRACE_LOGGER.addHandler(_handler)
TRACE_LOGGER.setLevel(logging.INFO)
TRACE_LOGGER.propagate = False


class PipelineTrace:
    """Emit step-by-step terminal logs with a stable trace id.

    Every line — START, a step (success/failure), info, warning, error,
    DONE/FAILED — shares one marker+name column layout so the "|" field
    lists all start on the same screen column: 🟢 start, ✅ step success /
    DONE, ❌ step failure / error() / FAILED, 🔷 info, 🔶 warning. The
    marker is always a single emoji glyph, since those render double-width
    in terminals (unlike a 2-space blank of plain text, which only looks
    aligned by coincidence at this exact width; a longer literal like
    "DONE   " drifts the rest of the line out of alignment entirely).
    """

    EVENT_WIDTH = 28
    _MARK_OK = "✅"
    _MARK_FAIL = "❌"
    _MARK_INFO = "🔷"
    _MARK_WARNING = "⚠️"
    _MARK_START = "🟢"

    def __init__(
        self,
        pipeline: str,
        trace_id: str,
        on_step: Callable[[str], None] | None = None,
        **fields: Any,
    ) -> None:
        self.pipeline = pipeline
        self.trace_id = trace_id
        self.fields = fields
        self.started = time.perf_counter()
        self.step_no = 0
        self.on_step = on_step

    def start(self, **fields: Any) -> None:
        TRACE_LOGGER.info(
            self._line(self._MARK_START, "START", {**self.fields, **fields})
        )

    def info(self, event: str, *, icon: str | None = None, **fields: Any) -> None:
        # icon lets a caller pick an event-specific glyph instead of the
        # generic marker, e.g. distinguishing "image described" from
        # "category classified" at a glance. Must be a single emoji glyph
        # (see the class docstring) to keep the column aligned.
        TRACE_LOGGER.info(self._line(icon or self._MARK_INFO, event, fields))

    def warning(self, event: str, **fields: Any) -> None:
        TRACE_LOGGER.warning(self._line(self._MARK_WARNING, event, fields))

    def error(self, event: str, **fields: Any) -> None:
        TRACE_LOGGER.error(self._line(self._MARK_FAIL, event, fields))

    def complete(self, **fields: Any) -> None:
        elapsed = time.perf_counter() - self.started
        TRACE_LOGGER.info(
            self._line(
                self._MARK_OK,
                "DONE",
                {**self.fields, "duration": self.format_duration(elapsed), **fields},
            )
        )

    def fail(self, exc: BaseException, **fields: Any) -> None:
        elapsed = time.perf_counter() - self.started
        TRACE_LOGGER.error(
            self._line(
                self._MARK_FAIL,
                "FAILED",
                {
                    **self.fields,
                    "duration": self.format_duration(elapsed),
                    "error_type": type(exc).__name__,
                    **fields,
                },
            )
        )

    @contextmanager
    def step(self, name: str, **fields: Any):
        self.step_no += 1
        step_fields = dict(fields)
        if self.on_step:
            self.on_step(name)
        started = time.perf_counter()
        try:
            yield step_fields
        except Exception as exc:
            elapsed = time.perf_counter() - started
            TRACE_LOGGER.exception(
                self._line(
                    self._MARK_FAIL,
                    name,
                    {
                        "duration": self.format_duration(elapsed),
                        **step_fields,
                        "error_type": type(exc).__name__,
                    },
                )
            )
            raise
        else:
            elapsed = time.perf_counter() - started
            TRACE_LOGGER.info(
                self._line(
                    self._MARK_OK,
                    name,
                    {"duration": self.format_duration(elapsed), **step_fields},
                )
            )

    def _prefix(self) -> str:
        return f"[{self.pipeline.upper()} {self.trace_id}]"

    def _line(self, mark: str, name: str, fields: dict[str, Any]) -> str:
        return (
            f"{self._prefix()} {mark} {name:<{self.EVENT_WIDTH}} "
            f"{self._format_fields(fields)}"
        )

    def _format_fields(self, fields: dict[str, Any]) -> str:
        parts = [
            f"{key}={self._format_value(value)}"
            for key, value in fields.items()
            if value is not None
        ]
        return "| " + " | ".join(parts) if parts else ""

    def format_duration(self, elapsed_seconds: float) -> str:
        elapsed_ms = round(elapsed_seconds * 1000)
        if elapsed_ms < 1000:
            return f"{elapsed_ms}ms"
        return f"{elapsed_seconds:.2f}s"

    def _format_value(self, value: Any) -> str:
        text = str(value).replace("\n", " ").strip()
        if len(text) > 120:
            text = text[:117] + "..."
        if any(ch.isspace() for ch in text):
            return repr(text)
        return text
