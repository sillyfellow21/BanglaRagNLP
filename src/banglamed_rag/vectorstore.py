from __future__ import annotations

from pathlib import Path
from typing import Any

import chromadb
from langchain_core.documents import Document

from .config import settings
from .embedder import BanglaEmbedder
from .utils import read_jsonl


class ChromaVectorStore:
    def __init__(
        self,
        embedder: BanglaEmbedder | None = None,
        persist_path: str | None = None,
        collection_name: str | None = None,
    ) -> None:
        self.embedder = embedder
        self.persist_path = persist_path or str(settings.chroma_path)
        self.collection_name = collection_name or settings.collection_name
        self.client = chromadb.PersistentClient(path=self.persist_path)
        self.collection = self.client.get_or_create_collection(self.collection_name)

    def create_collection(self, name: str | None = None, reset: bool = False) -> None:
        if name:
            self.collection_name = name
        if reset:
            try:
                self.client.delete_collection(self.collection_name)
            except Exception:
                pass
        self.collection = self.client.get_or_create_collection(self.collection_name)

    def add_documents(
        self,
        ids: list[str],
        texts: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]],
    ) -> None:
        if not ids:
            return
        self.collection.add(ids=ids, documents=texts, embeddings=embeddings, metadatas=metadatas)

    def query(self, query_embedding: list[float], n_results: int = 5) -> list[Document]:
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )
        documents: list[Document] = []
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        for doc, meta in zip(docs, metas, strict=True):
            documents.append(Document(page_content=doc, metadata=meta or {}))
        return documents


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
    ids = [record.get("id", "") for record in records]
    texts = [record.get("text", "") for record in records]
    metadatas = [
        {
            "title": record.get("title", ""),
            "url": record.get("url", ""),
            "source_id": record.get("source_id", ""),
            "source": record.get("source", ""),
            "chunk_index": record.get("chunk_index", ""),
        }
        for record in records
    ]
    embeddings = embedder.embed_documents(texts)
    store.add_documents(ids=ids, texts=texts, embeddings=embeddings, metadatas=metadatas)
    return store


def main() -> None:
    store = build_vectorstore(reset=True)
    count = store.collection.count()
    print(f"Vector store ready with {count} chunks.")


if __name__ == "__main__":
    main()
