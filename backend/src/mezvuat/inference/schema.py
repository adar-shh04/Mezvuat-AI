"""JSON schemas in the portable subset (contract §2.5, ADR-0011).

Rules: every property required, `additionalProperties: false`, nullable written as `["T", "null"]`, no `$ref`,
`oneOf`, `anyOf`, `allOf`, `pattern`, `minLength`, `maxLength` (or other numeric/length keywords), depth at most 5.
Length and count limits live in the Pydantic model and are enforced after parsing.
"""

from __future__ import annotations

import copy
from typing import Any

from pydantic import BaseModel

MAX_DEPTH = 5
_FORBIDDEN = frozenset(
    {
        "$ref",
        "$defs",
        "oneOf",
        "anyOf",
        "allOf",
        "not",
        "pattern",
        "minLength",
        "maxLength",
        "minItems",
        "maxItems",
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "format",
        "default",
        "title",
        "discriminator",
        "const",
    }
)


def _resolve(node: Any, defs: dict[str, Any]) -> Any:
    if isinstance(node, list):
        return [_resolve(item, defs) for item in node]
    if not isinstance(node, dict):
        return node
    if "$ref" in node:
        name = node["$ref"].rsplit("/", 1)[-1]
        merged = {**copy.deepcopy(defs[name]), **{k: v for k, v in node.items() if k != "$ref"}}
        return _resolve(merged, defs)
    if "anyOf" in node:
        options = [_resolve(option, defs) for option in node["anyOf"]]
        non_null = [o for o in options if o.get("type") != "null"]
        has_null = len(non_null) < len(options)
        if len(non_null) != 1:
            raise ValueError("only `T | None` unions are portable")
        inner = dict(non_null[0])
        if has_null:
            inner_type = inner.get("type")
            if not isinstance(inner_type, str):
                raise ValueError("nullable types must have a single base type")
            inner["type"] = [inner_type, "null"]
        extras = {k: v for k, v in node.items() if k not in ("anyOf",)}
        return {**inner, **extras}
    if "const" in node and "type" in node:
        node = {**node, "enum": [node["const"]]}
    return {key: _resolve(value, defs) for key, value in node.items()}


def _strip(node: Any) -> Any:
    if isinstance(node, list):
        return [_strip(item) for item in node]
    if not isinstance(node, dict):
        return node
    cleaned: dict[str, Any] = {}
    for key, value in node.items():
        if key in _FORBIDDEN:
            continue
        if key == "properties":
            cleaned[key] = {name: _strip(prop) for name, prop in value.items()}
        else:
            cleaned[key] = _strip(value)
    if cleaned.get("type") == "object" or "properties" in cleaned:
        cleaned["type"] = "object"
        cleaned["required"] = list(cleaned.get("properties", {}))
        cleaned["additionalProperties"] = False
    return cleaned


def portable_schema(model: type[BaseModel]) -> dict[str, Any]:
    raw = model.model_json_schema()
    defs = raw.pop("$defs", {})
    schema: dict[str, Any] = _strip(_resolve(raw, defs))
    problems = portability_problems(schema)
    if problems:
        raise ValueError(f"{model.__name__} is not portable: {problems}")
    return schema


def portability_problems(schema: dict[str, Any]) -> list[str]:
    problems: list[str] = []

    def walk(node: Any, path: str, depth: int) -> None:
        if isinstance(node, list):
            for index, item in enumerate(node):
                walk(item, f"{path}[{index}]", depth)
            return
        if not isinstance(node, dict):
            return
        for key in node:
            if key in _FORBIDDEN:
                problems.append(f"{path}: forbidden keyword {key}")
        if node.get("type") == "object" or "properties" in node:
            depth += 1
            if depth > MAX_DEPTH:
                problems.append(f"{path}: nesting deeper than {MAX_DEPTH}")
            props = node.get("properties", {})
            if node.get("additionalProperties") is not False:
                problems.append(f"{path}: additionalProperties must be false")
            if sorted(node.get("required", [])) != sorted(props):
                problems.append(f"{path}: every property must be required")
            for name, prop in props.items():
                walk(prop, f"{path}.{name}", depth)
        if "items" in node:
            walk(node["items"], f"{path}[]", depth)

    walk(schema, "$", 0)
    return problems
