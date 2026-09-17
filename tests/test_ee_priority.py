"""Unit tests for the shared SITS-over-imagery priority gate.

Pure asyncio — no Earth Engine, no cache/DB involved.
"""

from __future__ import annotations

import asyncio

import pytest

from agrigee_lite.ee_priority import PriorityGate


def test_capacity_is_enforced() -> None:
    async def run() -> None:
        gate = PriorityGate(capacity=2)
        order: list[str] = []

        async def worker(name: str) -> None:
            async with gate.priority(high_priority=False):
                order.append(f"{name}-start")
                await asyncio.sleep(0.02)
                order.append(f"{name}-end")

        await asyncio.gather(worker("a"), worker("b"), worker("c"))

        # Never more than 2 concurrently "started" without an "end" between.
        active = 0
        for event in order:
            if event.endswith("-start"):
                active += 1
            else:
                active -= 1
            assert active <= 2

    asyncio.run(run())


def test_high_priority_jumps_ahead_of_queued_low_priority() -> None:
    """A SITS (high) acquire arriving after imagery (low) waiters are already
    queued must still be granted the next free slot before them."""

    async def run() -> None:
        gate = PriorityGate(capacity=1)
        order: list[str] = []
        release_first = asyncio.Event()

        async def holder() -> None:
            async with gate.priority(high_priority=False):
                order.append("holder-start")
                await release_first.wait()
            order.append("holder-end")

        async def low(name: str) -> None:
            async with gate.priority(high_priority=False):
                order.append(name)

        async def high(name: str) -> None:
            async with gate.priority(high_priority=True):
                order.append(name)

        holder_task = asyncio.create_task(holder())
        await asyncio.sleep(0)  # let holder grab the only slot

        low_task = asyncio.create_task(low("low-1"))
        await asyncio.sleep(0)  # low-1 is now queued, waiting
        high_task = asyncio.create_task(high("high-1"))
        await asyncio.sleep(0)  # high-1 queued after low-1, but higher priority

        release_first.set()
        await asyncio.gather(holder_task, low_task, high_task)

        assert order.index("high-1") < order.index("low-1")

    asyncio.run(run())


def test_low_priority_cannot_skip_ahead_of_waiting_high_priority() -> None:
    """Even a low-priority acquire that arrives while a slot is free must
    defer to an already-queued high-priority waiter."""

    async def run() -> None:
        gate = PriorityGate(capacity=1)
        order: list[str] = []
        release_first = asyncio.Event()

        async def holder() -> None:
            async with gate.priority(high_priority=True):
                await release_first.wait()

        async def high(name: str) -> None:
            async with gate.priority(high_priority=True):
                order.append(name)

        async def low(name: str) -> None:
            async with gate.priority(high_priority=False):
                order.append(name)

        holder_task = asyncio.create_task(holder())
        await asyncio.sleep(0)

        high_task = asyncio.create_task(high("high-1"))
        await asyncio.sleep(0)

        release_first.set()
        # Low priority submitted only after high-1 already queued.
        low_task = asyncio.create_task(low("low-1"))
        await asyncio.gather(holder_task, high_task, low_task)

        assert order.index("high-1") < order.index("low-1")

    asyncio.run(run())


def test_cancelled_waiter_does_not_leak_capacity() -> None:
    async def run() -> None:
        gate = PriorityGate(capacity=1)
        release_first = asyncio.Event()

        async def holder() -> None:
            async with gate.priority(high_priority=False):
                await release_first.wait()

        async def waiter() -> None:
            async with gate.priority(high_priority=False):
                pass  # pragma: no cover - expected to be cancelled first

        holder_task = asyncio.create_task(holder())
        await asyncio.sleep(0)
        waiter_task = asyncio.create_task(waiter())
        await asyncio.sleep(0)

        waiter_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter_task

        release_first.set()
        await holder_task

        # A brand new acquire must be able to get the slot immediately —
        # nothing left behind by the cancelled waiter.
        acquired = False

        async def probe() -> None:
            nonlocal acquired
            async with gate.priority(high_priority=False):
                acquired = True

        await asyncio.wait_for(probe(), timeout=1)
        assert acquired

    asyncio.run(run())
