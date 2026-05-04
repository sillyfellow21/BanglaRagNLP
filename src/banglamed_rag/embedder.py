from __future__ import annotations

from typing import Iterable

from sentence_transformers import SentenceTransformer

from .config import settings


class BanglaEmbedder:
    def __init__(self, model_name: str | None = None, device: str | None = None) -> None:
        self.model_name = model_name or settings.embed_model
        self.model = SentenceTransformer(self.model_name, device=device)
        self.dim = int(self.model.get_sentence_embedding_dimension())

    def embed_documents(self, texts: Iterable[str]) -> list[list[float]]:
        embeddings = self.model.encode(list(texts), normalize_embeddings=True)
        return [list(map(float, embedding)) for embedding in embeddings]

    def embed_query(self, text: str) -> list[float]:
        embedding = self.model.encode(text, normalize_embeddings=True)
        return list(map(float, embedding))
