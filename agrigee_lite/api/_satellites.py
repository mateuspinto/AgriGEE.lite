"""
Satellite registry: maps class name → class, enabling satellite specification
via JSON ({"name": "Sentinel2", "params": {"use_sr": false}}).
"""

import inspect

import agrigee_lite.sat as _sat
from agrigee_lite.sat.abstract_satellite import AbstractSatellite

# Auto-discovered from agrigee_lite.sat — no manual maintenance needed
REGISTRY: dict[str, type[AbstractSatellite]] = {
    name: cls
    for name, cls in inspect.getmembers(_sat, inspect.isclass)
    if issubclass(cls, AbstractSatellite) and cls is not AbstractSatellite
}


def build_satellite(name: str, params: dict) -> AbstractSatellite:
    cls = REGISTRY.get(name)
    if cls is None:
        available = sorted(REGISTRY)
        raise ValueError(f"Unknown satellite '{name}'. Available: {available}")
    return cls(**params)
