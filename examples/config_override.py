from __future__ import annotations

import json
import os
from dataclasses import asdict, fields
from typing import Any, TypeVar, get_args, get_origin

T = TypeVar("T")


def _coerce_value(value: Any, annotation: Any) -> Any:
    origin = get_origin(annotation)
    if origin is tuple and isinstance(value, (list, tuple)):
        args = get_args(annotation)
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_coerce_value(item, args[0]) for item in value)
        if args and len(args) == len(value):
            return tuple(_coerce_value(item, item_type) for item, item_type in zip(value, args))
        return tuple(value)
    return value


def load_dataclass_override(env_var: str, default_config: T, config_type: type[T]) -> T:
    raw = os.environ.get(env_var)
    if not raw:
        return default_config

    payload = json.loads(raw)
    merged = asdict(default_config)
    merged.update(payload)
    field_types = {field.name: field.type for field in fields(config_type)}
    coerced = {
        name: _coerce_value(value, field_types.get(name))
        for name, value in merged.items()
    }
    return config_type(**coerced)
