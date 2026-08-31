"""In-memory log monitor backing the live dashboard at ``GET /monitor``.

Installs a single :class:`logging.Handler` on the root logger that captures
every log record emitted anywhere in the process (agrigee_lite modules,
uvicorn, etc.) into a small ring buffer, tagging each one with a display
``category`` (success / error / warning / debug / info) used to color the
dashboard. Call sites can force a category explicitly::

    logger.info("Downloaded %d rows", n, extra={"agl_category": "success"})

Without an explicit ``agl_category``, the category falls back to the
record's level (DEBUG -> debug, WARNING -> warning, ERROR/CRITICAL -> error,
everything else -> info).

The buffer and its subscriber queues live in module-level state so every
request handler (and the MCP sidecar's own FastAPI app instance) shares the
same view of what the API process is doing.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal

from agrigee_lite.config import MONITOR_MAX_LOG_ENTRIES

Category = Literal["success", "error", "warning", "debug", "info"]
_CATEGORIES: tuple[Category, ...] = ("success", "error", "warning", "debug", "info")


@dataclass(slots=True)
class LogEntry:
    seq: int
    ts: str
    level: str
    category: Category
    logger: str
    message: str

    def to_dict(self) -> dict[str, str | int]:
        return {
            "seq": self.seq,
            "ts": self.ts,
            "level": self.level,
            "category": self.category,
            "logger": self.logger,
            "message": self.message,
        }


class MonitorState:
    """Bounded ring buffer of log entries plus live SSE subscribers."""

    def __init__(self, maxlen: int = MONITOR_MAX_LOG_ENTRIES) -> None:
        self._entries: deque[LogEntry] = deque(maxlen=maxlen)
        self._seq = 0
        self._counters: dict[Category, int] = dict.fromkeys(_CATEGORIES, 0)
        self._subscribers: set[asyncio.Queue[LogEntry]] = set()

    def add(self, level: str, category: Category, logger_name: str, message: str) -> None:
        self._seq += 1
        entry = LogEntry(
            seq=self._seq,
            ts=datetime.now(UTC).isoformat(timespec="seconds"),
            level=level,
            category=category,
            logger=logger_name,
            message=message,
        )
        self._entries.append(entry)
        self._counters[category] += 1
        for queue in list(self._subscribers):
            with contextlib.suppress(asyncio.QueueFull):
                queue.put_nowait(entry)

    def entries_after(self, after_seq: int) -> list[LogEntry]:
        return [e for e in self._entries if e.seq > after_seq]

    def snapshot(self) -> dict:
        return {
            "entries": [e.to_dict() for e in self._entries],
            "counters": dict(self._counters),
            "last_seq": self._seq,
        }

    def subscribe(self) -> asyncio.Queue[LogEntry]:
        queue: asyncio.Queue[LogEntry] = asyncio.Queue(maxsize=500)
        self._subscribers.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue[LogEntry]) -> None:
        self._subscribers.discard(queue)


monitor_state = MonitorState()


def _categorize(record: logging.LogRecord) -> Category:
    explicit = getattr(record, "agl_category", None)
    if explicit in _CATEGORIES:
        return explicit
    if record.levelno >= logging.ERROR:
        return "error"
    if record.levelno >= logging.WARNING:
        return "warning"
    if record.levelno <= logging.DEBUG:
        return "debug"
    return "info"


class MonitorLogHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = record.getMessage()
            if record.exc_info:
                message = f"{message}\n{self.formatException(record.exc_info)}"
        except Exception:
            message = f"<unformattable log record: {record.msg!r}>"
        monitor_state.add(record.levelname, _categorize(record), record.name, message)


_installed = False


def install_monitor_handler() -> None:
    """Attach the monitor handler to the root logger (idempotent).

    Only the ``agrigee_lite`` logger tree is bumped to DEBUG so its own
    debug-level messages (cache hits, per-chunk progress) reach the
    dashboard; third-party loggers (aiohttp, uvicorn, ...) keep their
    default level, so their debug noise never reaches the shared handler.
    """
    global _installed
    if _installed:
        return
    handler = MonitorLogHandler()
    handler.setLevel(logging.DEBUG)
    logging.getLogger().addHandler(handler)
    logging.getLogger("agrigee_lite").setLevel(logging.DEBUG)
    _installed = True
