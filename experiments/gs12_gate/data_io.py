"""Immutable, hash-checked GS-12 JSON and trajectory artifacts."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def immutable_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = canonical(value)+"\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise FileExistsError(f"existing artifact differs: {path}")
        return file_hash(path)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=".gs12_",
                                     encoding="utf-8", delete=False) as stream:
        stream.write(content)
        temporary = Path(stream.name)
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return file_hash(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_npz(path, **arrays):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".gs12_",
                                     suffix=".npz", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        np.savez_compressed(temporary, **arrays)
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return file_hash(path)


def read_cases(path):
    cases = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    ids = [case["case_id"] for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate case_id")
    return cases


def immutable_jsonl(path, records):
    path = Path(path)
    content = "".join(canonical(record)+"\n" for record in records)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise FileExistsError(f"existing case manifest differs: {path}")
    else:
        path.write_text(content, encoding="utf-8")
    return file_hash(path)
