"""Small helpers: JSONL files, text normalisation, stopwords."""

import json
import re
from typing import Any
import unicodedata
from functools import cache
from pathlib import Path
from probad.constants import BN_STOPWORD_PATH
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


PUNCT = "\"'“”‘’.,!?;:()[]{}।॥—–-…"  # stripped from word edges (includes the Bengali danda)


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalise_text(text, lower=True):
    """NFC-normalise, collapse whitespace, optionally lowercase."""
    text = re.sub(r"\s+", " ", unicodedata.normalize("NFC", text)).strip()
    return text.lower() if lower else text


def words(text, lower=True):
    """Whitespace tokens with punctuation stripped (works for Bengali too)."""
    return [w for w in (t.strip(PUNCT) for t in normalise_text(text, lower).split(" ")) if w]


def content_words(text, stopwords, lower=True):
    return [w for w in words(text, lower) if w not in stopwords]


@cache
def load_english_stopwords() ->frozenset:
    return frozenset(ENGLISH_STOP_WORDS)


@cache
def load_bengali_stopwords() -> frozenset:
    source = Path(__file__).parent / BN_STOPWORD_PATH
    # source = Path(BN_STOPWORD_PATH)

    try:
        data: Any = json.loads(source.read_text(encoding="utf-8-sig"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Stopword file not found: {source}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid JSON in stopword file {source}: "
            f"line {exc.lineno}, column {exc.colno}: {exc.msg}"
        ) from exc
    
    return frozenset([word.strip() for word in data])

def print_div(text: str = "", w: int = 100) -> None:
    if not text:
            print("=" * w)
    else:
        text = f"> {text} <"
        print(text.center(w, "="))