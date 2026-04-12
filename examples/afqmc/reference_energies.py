from __future__ import annotations

import json
from pathlib import Path


REFERENCE_FILE = Path(__file__).resolve().parent / "reference_energies.json"
PRIMARY_REFERENCE_KEY = "primary_reference"
REFERENCE_SOURCES_KEY = "references"


def load_reference_data() -> dict:
    if not REFERENCE_FILE.exists():
        return {}
    text = REFERENCE_FILE.read_text().strip()
    if not text:
        return {}
    return json.loads(text)


def _record(system: str) -> dict | None:
    payload = load_reference_data()
    record = payload.get(system)
    if not isinstance(record, dict):
        return None
    return record


def _normalize_method_key(method: str) -> str:
    return method.strip().lower().replace("-", "_").replace(" ", "_").replace("(", "").replace(")", "")


def reference_sources(system: str) -> dict[str, dict]:
    record = _record(system)
    if record is None:
        return {}
    payload = record.get(REFERENCE_SOURCES_KEY)
    if not isinstance(payload, dict):
        return {}
    sources = {}
    for key, value in payload.items():
        if not isinstance(value, dict):
            continue
        source = dict(value)
        source.setdefault("method_key", str(key))
        sources[str(key)] = source
    return sources


def _primary_reference_key(record: dict, sources: dict[str, dict]) -> str | None:
    preferred = record.get(PRIMARY_REFERENCE_KEY)
    if isinstance(preferred, str) and preferred in sources:
        return preferred
    method = record.get("reference_method")
    if isinstance(method, str):
        normalized = _normalize_method_key(method)
        if normalized in sources:
            return normalized
    if not sources:
        return None
    return next(iter(sources))


def reference_record(system: str, method: str | None = None) -> dict | None:
    record = _record(system)
    if record is None:
        return None
    if method is None:
        return record
    sources = reference_sources(system)
    key = _normalize_method_key(method)
    source = sources.get(key)
    if source is None:
        return None
    return source


def reference_energy(system: str, method: str | None = None) -> float | None:
    record = _record(system)
    if record is None:
        return None
    if method is not None:
        source = reference_record(system, method)
        if source is None or source.get("reference_energy") is None:
            return None
        return float(source["reference_energy"])
    sources = reference_sources(system)
    primary_key = _primary_reference_key(record, sources)
    if primary_key is not None and sources[primary_key].get("reference_energy") is not None:
        return float(sources[primary_key]["reference_energy"])
    return None


def reference_method(system: str, method: str | None = None) -> str | None:
    if method is not None:
        record = reference_record(system, method)
        if record is None:
            return None
        label = record.get("reference_method")
        return None if label is None else str(label)
    record = _record(system)
    if record is None:
        return None
    sources = reference_sources(system)
    primary_key = _primary_reference_key(record, sources)
    if primary_key is not None:
        label = sources[primary_key].get("reference_method")
        if label is not None:
            return str(label)
    return None


def reference_stderr(system: str, method: str | None = None) -> float | None:
    if method is None:
        record = _record(system)
        if record is None:
            return None
        sources = reference_sources(system)
        primary_key = _primary_reference_key(record, sources)
        if primary_key is not None:
            source = sources[primary_key]
            if source.get("stderr") is not None:
                return float(source["stderr"])
        return None
    source = reference_record(system, method)
    if source is None or source.get("stderr") is None:
        return None
    return float(source["stderr"])
