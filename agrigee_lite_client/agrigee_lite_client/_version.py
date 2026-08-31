"""Single source of truth for this package's version at import time.

Kept separate from __init__.py so _client.py/_async_client.py can import it
without a circular import (they're imported BY __init__.py). Must always
equal pyproject.toml's version (SPECS.md §2.1) and, transitively, the
server's — see also the embedded satellite date table in _satellite_dates.py,
which is only valid when client and server versions agree.
"""

__version__ = "3.5.0"
