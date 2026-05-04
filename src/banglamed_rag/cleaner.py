from __future__ import annotations

import re
from typing import Iterable

from langdetect import DetectorFactory, detect

from .config import settings
from .utils import has_bengali, normalize_whitespace, read_jsonl, strip_urls_emails, write_jsonl

DetectorFactory.seed = 0

SENTENCE_SPLIT_RE = re.compile(r"(?<=[।!?])\s+")

try:
    from indicnlp.tokenize import sentence_tokenize

    def split_sentences(text: str) -> list[str]:
        return sentence_tokenize.sentence_split(text, lang="bn")

except Exception:

    def split_sentences(text: str) -> list[str]:
        return [s.strip() for s in SENTENCE_SPLIT_RE.split(text) if s.strip()]


def is_english(text: str) -> bool:
    try:
        return detect(text) == "en"
    except Exception:
        return False


def clean_text(text: str) -> str:
    text = strip_urls_emails(text)
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    cleaned: list[str] = []
    for paragraph in paragraphs:
        if is_english(paragraph):
            continue
        if not has_bengali(paragraph):
            continue
        cleaned.append(paragraph)
    return normalize_whitespace("\n".join(cleaned))


def chunk_sentences(
    sentences: Iterable[str], max_tokens: int, overlap: int
) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0
    sentence_list = [s for s in sentences if s.strip()]
    idx = 0
    while idx < len(sentence_list):
        sentence = sentence_list[idx]
        token_count = len(sentence.split())
        if not current or current_tokens + token_count <= max_tokens:
            current.append(sentence)
            current_tokens += token_count
            idx += 1
            continue
        chunks.append(" ".join(current).strip())
        overlap_tokens = 0
        overlap_sentences: list[str] = []
        for sent in reversed(current):
            overlap_sentences.insert(0, sent)
            overlap_tokens += len(sent.split())
            if overlap_tokens >= overlap:
                break
        current = overlap_sentences
        current_tokens = sum(len(sent.split()) for sent in current)
    if current:
        chunks.append(" ".join(current).strip())
    return chunks


def clean_corpus(
    input_path=settings.corpus_path,
    output_path=settings.cleaned_corpus_path,
    max_tokens: int | None = None,
    overlap: int | None = None,
) -> list[dict[str, str]]:
    records = read_jsonl(input_path)
    cleaned_records: list[dict[str, str]] = []
    max_tokens = max_tokens or settings.chunk_size
    overlap = overlap or settings.chunk_overlap
    for record in records:
        raw_text = record.get("text", "")
        cleaned_text = clean_text(raw_text)
        if not cleaned_text:
            continue
        sentences = split_sentences(cleaned_text)
        chunks = chunk_sentences(sentences, max_tokens=max_tokens, overlap=overlap)
        for idx, chunk in enumerate(chunks):
            cleaned_records.append(
                {
                    "id": f"{record.get('id', 'doc')}_{idx}",
                    "source_id": record.get("id", ""),
                    "title": record.get("title", ""),
                    "url": record.get("url", ""),
                    "text": chunk,
                    "chunk_index": str(idx),
                }
            )
    write_jsonl(output_path, cleaned_records)
    return cleaned_records


def main() -> None:
    records = clean_corpus()
    print(f"Saved {len(records)} chunks to {settings.cleaned_corpus_path}")


if __name__ == "__main__":
    main()
