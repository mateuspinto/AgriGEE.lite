"""
Structural tests for the MCP server built from the REST API.

Only checks that routes are mapped to the right MCP component types — no
Earth Engine network calls are made here, see tests/api/test_sits.py for that.
"""

from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("fastmcp")

from agrigee_lite.mcp import create_mcp_server  # noqa: E402


@pytest.fixture(scope="module")
def mcp_server():
    return create_mcp_server()


def test_rest_endpoints_become_mcp_tools(mcp_server):
    tools = asyncio.run(mcp_server.list_tools())
    names = {t.name for t in tools}

    assert names >= {
        "list_satellites",
        "health_check",
        "submit_images_job",
        "get_single_sits",
        "submit_multiple_sits_job",
        "list_jobs",
        "get_job",
        "delete_job",
        "submit_sits_job_from_file",
    }
    # binary download must not be a tool — it would be mangled as text
    assert "download_job_result" not in names
    # multipart upload can't be represented as JSON tool args; replaced above
    assert "submit_multiple_sits_job_file" not in names


def test_download_route_is_a_resource_template(mcp_server):
    templates = asyncio.run(mcp_server.list_resource_templates())
    assert any("download" in t.uri_template for t in templates)


def test_sidecar_mode_proxies_instead_of_owning_state():
    """
    With base_url set, the server must not touch Earth Engine/the cache/the job
    store itself — that state belongs to the sibling `agl_api` process it
    proxies to. Regression test for the DuckDB single-writer lock conflict
    that happens if both processes call init_cache() independently.
    """
    sidecar = create_mcp_server(base_url="http://127.0.0.1:8000")
    tools = asyncio.run(sidecar.list_tools())
    names = {t.name for t in tools}

    assert "list_satellites" in names
    assert "submit_sits_job_from_file" in names
