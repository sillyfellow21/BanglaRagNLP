from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

BENGALI_RE = re.compile(r"[\u0980-\u09FF]")
URL_RE = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
EMAIL_RE = re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b")
WHITESPACE_RE = re.compile(r"\s+")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def slugify(text: str) -> str:
    cleaned = re.sub(r"[^\w\u0980-\u09FF-]+", "_", text, flags=re.UNICODE).strip("_")
    return cleaned or "doc"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def strip_urls_emails(text: str) -> str:
    text = URL_RE.sub(" ", text)
    text = EMAIL_RE.sub(" ", text)
    return text


def normalize_whitespace(text: str) -> str:
    return WHITESPACE_RE.sub(" ", text).strip()


def has_bengali(text: str) -> bool:
    return bool(BENGALI_RE.search(text))
