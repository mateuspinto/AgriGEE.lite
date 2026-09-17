"""Offline regression for the actual fetch functions' semaphore admission.

Extract function definitions without importing the geospatial stack or initializing
Earth Engine. Tasks are cancelled while waiting for admission, before any I/O.
Run: python -m pytest --noconftest -q tests/test_fetch_admission.py
"""
import ast
import asyncio
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _gate_class():
    tree = ast.parse((ROOT / "agrigee_lite/ee_priority.py").read_text(encoding="utf-8"))
    # The default instance imports package config; only that construction is
    # omitted. The PriorityGate implementation itself executes unchanged.
    tree.body = [node for node in tree.body if not isinstance(node, ast.Assign)]
    namespace = {}
    exec(compile(tree, "ee_priority.py", "exec"), namespace)
    return namespace["PriorityGate"]


def _fetch_function(path, name, namespace):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    function = next(node for node in ast.walk(tree) if isinstance(node, ast.AsyncFunctionDef) and node.name == name)
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), function], type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, path, "exec"), namespace)
    return namespace[name]


@pytest.mark.parametrize("kind", ["image", "sits"])
def test_local_waiters_leave_global_capacity_available(kind):
    async def run():
        gate = _gate_class()(capacity=1)
        local = asyncio.Semaphore(0)
        namespace = {"EE_FETCH_GATE": gate, "semaphore": local}
        if kind == "image":
            fetch = _fetch_function("agrigee_lite/get/image.py", "_fetch_and_download_image", namespace)
            task = asyncio.create_task(fetch(
                chunk_index=0, ee_expression=None, image_names=[], image_indexes=[],
                ee_geometry=None, session=None, output_dir=None, semaphore=local,
                max_retries_per_chunk=1,
            ))
        else:
            fetch = _fetch_function("agrigee_lite/get/sits.py", "fetch_with_retry", namespace)
            task = asyncio.create_task(fetch(None, 0))
        await asyncio.sleep(0)
        try:
            # Another ready SITS fetch must enter even though this job has a
            # backlog waiting on its own local concurrency allowance.
            await asyncio.wait_for(gate.acquire(high_priority=True), timeout=0.2)
            gate.release()
        finally:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        await asyncio.wait_for(gate.acquire(high_priority=True), timeout=0.2)
        gate.release()

    asyncio.run(run())
