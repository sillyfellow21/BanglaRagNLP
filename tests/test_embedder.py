import banglamed_rag.embedder as embedder_module
from banglamed_rag.embedder import BanglaEmbedder


class DummyModel:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def get_sentence_embedding_dimension(self) -> int:
        return 768

    def encode(self, texts, normalize_embeddings: bool = True):
        if isinstance(texts, str):
            return [0.0] * 768
        return [[0.0] * 768 for _ in texts]


def test_embedder_dimension(monkeypatch) -> None:
    monkeypatch.setattr(embedder_module, "SentenceTransformer", DummyModel)
    embedder = BanglaEmbedder(model_name="dummy")
    doc_vectors = embedder.embed_documents(["এক", "দুই"])
    query_vector = embedder.embed_query("প্রশ্ন")
    assert len(doc_vectors[0]) == 768
    assert len(query_vector) == 768
