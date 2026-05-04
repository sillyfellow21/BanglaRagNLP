from banglamed_rag.vectorstore import ChromaVectorStore


class DummyEmbedder:
    def embed_documents(self, texts):
        return [[0.1, 0.1, 0.1] for _ in texts]

    def embed_query(self, text):
        return [0.1, 0.1, 0.1]


def test_vectorstore_roundtrip(tmp_path) -> None:
    store = ChromaVectorStore(
        DummyEmbedder(),
        persist_path=str(tmp_path),
        collection_name="test_collection",
    )
    store.add_documents(
        [
            {
                "id": "1",
                "text": "বাংলা চিকিৎসা",
                "title": "নমুনা",
                "url": "https://example.com",
                "source_id": "1",
            }
        ]
    )
    result = store.query("বাংলা", n_results=1)
    assert result.documents[0][0] == "বাংলা চিকিৎসা"
