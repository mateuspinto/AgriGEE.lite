"""Fixtures for API integration tests."""

from __future__ import annotations

import os
import pathlib
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import pytest
from fastapi import FastAPI  # pyright: ignore[reportMissingImports]
from fastapi.testclient import TestClient  # pyright: ignore[reportMissingImports]

import agrigee_lite.api as _api_module

# Load .env so GEE_KEY and other secrets are available (mirrors what `make api` does)
_env_file = pathlib.Path(".env")
if _env_file.exists():
    for _line in _env_file.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip())


def _make_patched_lifespan(db_path: pathlib.Path):
    @asynccontextmanager
    async def _lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        from agrigee_lite.api._jobs import job_store
        from agrigee_lite.cache import init_cache
        from agrigee_lite.ee_utils import ee_quick_start

        ee_quick_start()
        init_cache(db_path)
        job_store.load_from_db()
        yield

    return _lifespan


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    db_path = tmp_path_factory.mktemp("api_cache") / "test.duckdb"
    original = _api_module._lifespan
    _api_module._lifespan = _make_patched_lifespan(db_path)
    app = _api_module.create_app()
    with TestClient(app) as c:
        yield c
    _api_module._lifespan = original
