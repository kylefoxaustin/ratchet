"""Measurement-alias resolution helper.

Thin re-export point for resolve_measurement_key so surfaces can import alias
resolution from a dedicated module (the implementation lives with the schema in
model.py to avoid a circular dependency)."""
from ratchet.catalog.model import LLMModel, resolve_measurement_key

__all__ = ["resolve_measurement_key", "LLMModel"]
