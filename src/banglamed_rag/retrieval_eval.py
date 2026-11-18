"""Retrieval evaluation on the Bengali QA benchmark.

Each benchmark item has a question and the gold context passage that answers
it. All gold passages form the search corpus, and every question is scored on
whether its own passage is retrieved. This measures the retriever on its own,
without needing a local LLM, so it can run anywhere.

Usage:
    python -m banglamed_rag.retrieval_eval
    python -m banglamed_rag.retrieval_eval --models sagorsarker/bangla-bert-base
"""

from __future__ import annotations

import argparse
import math
import re
from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from .config import settings
from .utils import load_json, save_json

DEFAULT_MODELS = [
    "sagorsarker/bangla-bert-base",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
]
TOKEN_RE = re.compile(r"[ঀ-৿\w]+", re.UNICODE)

Ranker = Callable[[str], list[int]]


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def score_rankings(rankings: Sequence[Sequence[int]], ks: Sequence[int] = (1, 3, 5)) -> dict[str, float]:
    """Score rankings where query i's correct passage is passage i."""
    if not rankings:
        raise ValueError("no rankings to score")
    metrics: dict[str, float] = {}
    for k in ks:
        hits = sum(1 for i, ranking in enumerate(rankings) if i in ranking[:k])
        metrics[f"hit@{k}"] = hits / len(rankings)
    reciprocal = [1 / (list(ranking).index(i) + 1) if i in ranking else 0.0 for i, ranking in enumerate(rankings)]
    metrics["mrr"] = sum(reciprocal) / len(rankings)
    return metrics


def bm25_ranker(passages: Sequence[str], k1: float = 1.5, b: float = 0.75) -> Ranker:
    docs = [tokenize(p) for p in passages]
    avg_len = sum(len(d) for d in docs) / len(docs)
    df = Counter(term for d in docs for term in set(d))
    n = len(docs)
    idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
    freqs = [Counter(d) for d in docs]

    def rank(query: str) -> list[int]:
        scores = []
        for i, tf in enumerate(freqs):
            score = 0.0
            for term in tokenize(query):
                if term not in tf:
                    continue
                f = tf[term]
                score += idf[term] * f * (k1 + 1) / (f + k1 * (1 - b + b * len(docs[i]) / avg_len))
            scores.append(score)
        return sorted(range(n), key=lambda i: scores[i], reverse=True)

    return rank


def embedding_ranker(passages: Sequence[str], embed: Callable[[list[str]], list[list[float]]]) -> Ranker:
    """Rank passages by cosine similarity; `embed` must return normalised vectors."""
    passage_vecs = embed(list(passages))

    def rank(query: str) -> list[int]:
        q = embed([query])[0]
        scores = [sum(a * b for a, b in zip(q, vec, strict=True)) for vec in passage_vecs]
        return sorted(range(len(passage_vecs)), key=lambda i: scores[i], reverse=True)

    return rank


def evaluate_ranker(questions: Sequence[str], rank: Ranker) -> dict[str, float]:
    return score_rankings([rank(q) for q in questions])


def sentence_transformer_embed(model_name: str) -> Callable[[list[str]], list[list[float]]]:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)

    def embed(texts: list[str]) -> list[list[float]]:
        return [list(map(float, v)) for v in model.encode(texts, normalize_embeddings=True)]

    return embed


def run(models: Sequence[str]) -> dict[str, Any]:
    items = load_json(settings.qa_benchmark_path)
    questions = [item["question"] for item in items]
    passages = [item["context"] for item in items]

    results: dict[str, Any] = {"benchmark_size": len(items), "retrievers": {}}
    results["retrievers"]["bm25"] = evaluate_ranker(questions, bm25_ranker(passages))
    for name in models:
        ranker = embedding_ranker(passages, sentence_transformer_embed(name))
        results["retrievers"][name] = evaluate_ranker(questions, ranker)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--output", default=str(settings.data_dir / "retrieval_results.json"))
    args = parser.parse_args()

    results = run([m for m in args.models.split(",") if m])
    save_json(Path(args.output), results)
    print(f"Benchmark: {results['benchmark_size']} questions")
    for name, metrics in results["retrievers"].items():
        row = "  ".join(f"{k}={v:.3f}" for k, v in metrics.items())
        print(f"{name:65s} {row}")


if __name__ == "__main__":
    main()
