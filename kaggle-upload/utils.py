"""Small shared helpers: JSONL I/O, hashing, seeding, text normalisation, run log."""

from __future__ import annotations

import csv
import datetime as _dt
import hashlib
import json
import os
import random
import re
import unicodedata
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from probad.constants import PUNCT

PathLike = str | os.PathLike
BN_STOPWORDS_PATH = r"src/probad/stopwords-bn/stopwords-bn.json"

# --------------------------------------------------------------------------- JSONL


def read_jsonl(path: PathLike) -> list[dict[str, Any]]:
    """Read a JSONL file. Blank lines are skipped; errors report the line number."""
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:  # pragma: no cover - message path
                raise ValueError(f"{path}:{lineno}: invalid JSON ({e})") from e
    return rows


def iter_jsonl(path: PathLike) -> Iterator[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: PathLike, rows: Iterable[dict[str, Any]]) -> int:
    """Write rows to a JSONL file (UTF-8, Bengali kept readable). Returns row count."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def append_jsonl(path: PathLike, rows: Iterable[dict[str, Any]]) -> int:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_json(path: PathLike) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(path: PathLike, obj: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# --------------------------------------------------------------------------- hashing / time


def sha256_file(path: PathLike) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def now_iso() -> str:
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


def today() -> str:
    return _dt.date.today().isoformat()


# --------------------------------------------------------------------------- seeding


def set_seed(seed: int) -> None:
    """Seed python, numpy and (if installed) torch."""
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # pragma: no cover
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# --------------------------------------------------------------------------- text


_WS = re.compile(r"\s+")


def normalise_text(text: str, lower: bool = True) -> str:
    """NFC-normalise, collapse whitespace, optionally lowercase."""
    text = unicodedata.normalize("NFC", text)
    text = _WS.sub(" ", text).strip()
    return text.lower() if lower else text


def strip_punct(token: str) -> str:
    return token.strip(PUNCT)


def words(text: str, lower: bool = True) -> list[str]:
    """Whitespace tokenisation with punctuation stripped (works for Bengali too)."""
    out = []
    for tok in normalise_text(text, lower=lower).split(" "):
        tok = strip_punct(tok)
        if tok:
            out.append(tok)
    return out


def content_words(text: str, stopwords: frozenset[str], lower: bool = True) -> list[str]:
    return [w for w in words(text, lower=lower) if w not in stopwords]


# --------------------------------------------------------------------------- run log


RUN_LOG_FIELDS = [
    "timestamp",
    "kind",
    "model",
    "adapter",
    "condition",
    "seed",
    "data",
    "data_sha256",
    "config",
    "output",
    "summary",
    "notes",
]


def log_run(log_path: PathLike, **fields: Any) -> None:
    """Append one row to the experiment log CSV (results/log.csv by convention)."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    new = not log_path.exists()
    row = {k: "" for k in RUN_LOG_FIELDS}
    row["timestamp"] = now_iso()
    for k, v in fields.items():
        if k not in row:
            row["notes"] = (row["notes"] + f" {k}={v}").strip()
        else:
            row[k] = json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v
    with open(log_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=RUN_LOG_FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


def load_bengali_stopwords(path: str | Path = BN_STOPWORDS_PATH) -> set[str]:
    source = Path(path)

    try:
        data: Any = json.loads(source.read_text(encoding="utf-8-sig"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Stopword file not found: {source}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid JSON in stopword file {source}: "
            f"line {exc.lineno}, column {exc.colno}: {exc.msg}"
        ) from exc
    
    return {word.strip() for word in data}

def load_english_stopwords() -> set[str]:
    return set(ENGLISH_STOP_WORDS)

def flatten_dict(data: dict, *, separator: str = ".") -> dict:
    result = {}

    def _flatten(value, prefix=""):
        if isinstance(value, dict):
            for key, child in value.items():
                name = f"{prefix}{separator}{key}" if prefix else key
                _flatten(child, name)

        elif isinstance(value, list):
            for i, child in enumerate(value):
                _flatten(child, f"{prefix}{separator}{i}" if prefix else str(i))

        else:
            result[prefix] = value

    _flatten(data)
    return result