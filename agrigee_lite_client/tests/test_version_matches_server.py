"""Keeps agrigee_lite_client's version locked to agrigee_lite's. See SPECS.md §2.1.

Deliberately parses `version = "..."` with a regex instead of a full TOML
parser (tomllib is 3.11+, and this is the only place a TOML value is ever
read) — good enough for a file whose shape we control on both ends.
"""

from __future__ import annotations

import pathlib
import re

import agrigee_lite_client

_VERSION_RE = re.compile(r'^version\s*=\s*"([^"]+)"', re.MULTILINE)

_CLIENT_PYPROJECT = pathlib.Path(__file__).parent.parent / "pyproject.toml"
_SERVER_PYPROJECT = pathlib.Path(__file__).parent.parent.parent / "pyproject.toml"


def _read_version(pyproject_path: pathlib.Path) -> str:
    match = _VERSION_RE.search(pyproject_path.read_text())
    assert match is not None, f"no `version = \"...\"` found in {pyproject_path}"
    return match.group(1)


def test_pyproject_version_matches_server() -> None:
    assert _SERVER_PYPROJECT.exists(), (
        "server pyproject.toml not found — this test assumes agrigee_lite_client "
        "lives inside the AgriGEE.lite monorepo, see SPECS.md §2"
    )
    client_version = _read_version(_CLIENT_PYPROJECT)
    server_version = _read_version(_SERVER_PYPROJECT)
    assert client_version == server_version, (
        f"agrigee_lite_client ({client_version}) and agrigee_lite ({server_version}) "
        "versions have drifted — bump both in the same commit (SPECS.md §2.1)"
    )


def test_dunder_version_matches_pyproject() -> None:
    assert agrigee_lite_client.__version__ == _read_version(_CLIENT_PYPROJECT)
