from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chromadb

from .config import settings
from .embedder import BanglaEmbedder
from .utils import read_jsonl


@dataclass
class QueryResult:
    ids: list[list[str]]
    documents: list[list[str]]
    metadatas: list[list[dict[str, Any]]]
    distances: list[list[float]]


class ChromaVectorStore:
    def __init__(
        self,
        embedder: BanglaEmbedder,
        persist_path: str | None = None,
        collection_name: str | None = None,
    ) -> None:
        self.embedder = embedder
        self.persist_path = persist_path or str(settings.chroma_path)
        self.collection_name = collection_name or settings.collection_name
        self.client = chromadb.PersistentClient(path=self.persist_path)
        self.collection = self.client.get_or_create_collection(self.collection_name)

    def create_collection(self, reset: bool = False) -> None:
        if reset:
            self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(self.collection_name)

    def add_documents(self, records: list[dict[str, Any]]) -> None:
        if not records:
            return
        ids = [record.get("id") for record in records]
        documents = [record.get("text", "") for record in records]
        metadatas = [
            {
                "title": record.get("title", ""),
                "url": record.get("url", ""),
                "source_id": record.get("source_id", ""),
            }
            for record in records
        ]
        embeddings = self.embedder.embed_documents(documents)
        self.collection.add(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)

    def query(self, query_text: str, n_results: int = 5) -> QueryResult:
        embedding = self.embedder.embed_query(query_text)
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )
        return QueryResult(
            ids=results.get("ids", []),
            documents=results.get("documents", []),
            metadatas=results.get("metadatas", []),
            distances=results.get("distances", []),
        )


def build_vectorstore(
    corpus_path: Path | None = None,
    persist_path: str | None = None,
    reset: bool = False,
) -> ChromaVectorStore:
    corpus_path = corpus_path or settings.cleaned_corpus_path
    records = read_jsonl(corpus_path)
    embedder = BanglaEmbedder()
    store = ChromaVectorStore(embedder, persist_path=persist_path)
    store.create_collection(reset=reset)
    store.add_documents(records)
    return store


def main() -> None:
    store = build_vectorstore(reset=True)
    count = store.collection.count()
    print(f"Vector store ready with {count} chunks.")


if __name__ == "__main__":
    main()
