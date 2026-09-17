"""Global priority gate for Earth Engine fetch concurrency.

``POST /sits/multiple`` and ``POST /images`` each throttle their own chunk
downloads with a local, per-job semaphore (see ``get/sits.py``'s
``_AdaptiveSemaphore`` and ``get/image.py``'s plain ``asyncio.Semaphore``).
Those are independent: nothing stops an imagery job and a SITS job from
running at the same time, each opening up to its own limit of concurrent GEE
requests. Imagery downloads full-resolution multi-band GeoTIFFs per date —
far heavier per request than a SITS chunk — so a concurrent imagery job can
starve a SITS job of its share of the container's CPU/network budget even
though SITS is what a time-series chart is waiting on and imagery is only
ever a secondary, best-effort fetch.

``EE_FETCH_GATE`` is a single process-wide gate that both download paths
acquire around their actual GEE call, in addition to (not instead of) their
own local semaphore. Whenever a high-priority (SITS) waiter and a
low-priority (imagery) waiter are both queued for a slot, the SITS waiter is
always granted it first, regardless of arrival order. Capacity reuses
``EE_INTERACTIVE_CONCURRENCY`` (previously a documented-but-unused config
knob) since that is exactly what this gate now enforces.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections import deque
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager


class PriorityGate:
    """Fixed-capacity gate with two priority classes.

    A free slot always goes to the oldest waiting high-priority acquirer
    before any low-priority acquirer, even one that has been waiting longer.
    Within the same priority class, waiters are served in FIFO order.
    """

    def __init__(self, capacity: int) -> None:
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self._capacity = capacity
        self._in_use = 0
        self._high: deque[asyncio.Future[None]] = deque()
        self._low: deque[asyncio.Future[None]] = deque()

    def _next_queue(self) -> deque[asyncio.Future[None]] | None:
        if self._high:
            return self._high
        if self._low:
            return self._low
        return None

    def _wake_available(self) -> None:
        while self._in_use < self._capacity:
            queue = self._next_queue()
            if queue is None:
                return
            fut = queue.popleft()
            if fut.cancelled():
                continue
            self._in_use += 1
            fut.set_result(None)

    async def acquire(self, *, high_priority: bool) -> None:
        own_queue = self._high if high_priority else self._low
        # A low-priority caller may not jump a free slot while a high-priority
        # waiter is already queued for one.
        blocked_by_higher = not high_priority and bool(self._high)
        if self._in_use < self._capacity and not own_queue and not blocked_by_higher:
            self._in_use += 1
            return

        fut: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        own_queue.append(fut)
        try:
            await fut
        except asyncio.CancelledError:
            if fut.done() and not fut.cancelled():
                # Granted a slot right as the caller was cancelled - give it back.
                self._in_use -= 1
                self._wake_available()
            else:
                with contextlib.suppress(ValueError):  # already popped by _wake_available
                    own_queue.remove(fut)
            raise

    def release(self) -> None:
        self._in_use -= 1
        self._wake_available()

    @asynccontextmanager
    async def priority(self, *, high_priority: bool) -> AsyncIterator[None]:
        await self.acquire(high_priority=high_priority)
        try:
            yield
        finally:
            self.release()


def _make_default_gate() -> PriorityGate:
    from agrigee_lite.config import EE_INTERACTIVE_CONCURRENCY

    return PriorityGate(EE_INTERACTIVE_CONCURRENCY)


EE_FETCH_GATE = _make_default_gate()
