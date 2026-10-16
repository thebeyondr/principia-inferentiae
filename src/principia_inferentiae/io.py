"""Deterministic serialization and validated input loading."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TypeVar, cast

import yaml
from pydantic import BaseModel
from pydantic_core import to_jsonable_python

from principia_inferentiae.contracts import Digest, TraceRequest

ModelT = TypeVar("ModelT", bound=BaseModel)


def canonical_json_bytes(value: object) -> bytes:
    """Serialize without environmental or input-order dependence."""

    serializable = to_jsonable_python(value, by_alias=True, exclude_none=False)
    text = json.dumps(
        serializable,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return text.encode("utf-8")


def digest_value(value: object) -> Digest:
    return Digest(algorithm="sha256", value=hashlib.sha256(canonical_json_bytes(value)).hexdigest())


def digest_file(path: Path) -> Digest:
    return Digest(algorithm="sha256", value=hashlib.sha256(path.read_bytes()).hexdigest())


def load_document(path: Path) -> object:
    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    if suffix == ".json":
        return cast(object, json.loads(text))
    if suffix in {".yaml", ".yml"}:
        return cast(object, yaml.safe_load(text))
    raise ValueError(f"unsupported document extension: {path.suffix}")


def load_model(path: Path, model_type: type[ModelT]) -> ModelT:
    return model_type.model_validate(load_document(path))


def load_trace_jsonl(path: Path) -> tuple[TraceRequest, ...]:
    records: list[TraceRequest] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{line_number}: invalid JSON") from error
        records.append(TraceRequest.model_validate(raw))
    if not records:
        raise ValueError("trace must contain at least one request")
    request_ids = [record.request_id for record in records]
    if len(request_ids) != len(set(request_ids)):
        raise ValueError("trace request_id values must be unique")
    return tuple(records)


def write_json(path: Path, value: object) -> None:
    serializable = to_jsonable_python(value, by_alias=True, exclude_none=False)
    text = json.dumps(serializable, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{text}\n", encoding="utf-8")


def write_jsonl(path: Path, values: tuple[BaseModel, ...]) -> None:
    lines = []
    for value in values:
        serializable = to_jsonable_python(value, by_alias=True, exclude_none=False)
        lines.append(
            json.dumps(
                serializable,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_yaml(path: Path, value: object) -> None:
    serializable = to_jsonable_python(value, by_alias=True, exclude_none=False)
    text = yaml.safe_dump(
        serializable,
        allow_unicode=True,
        default_flow_style=False,
        sort_keys=False,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")
