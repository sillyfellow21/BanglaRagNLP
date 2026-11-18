from banglamed_rag.vectorstore import ChromaVectorStore


def test_vectorstore_roundtrip(tmp_path) -> None:
    store = ChromaVectorStore(
        embedder=None,
        persist_path=str(tmp_path),
        collection_name="test_collection",
    )
    ids = ["1"]
    texts = ["বাংলা চিকিৎসা"]
    embeddings = [[0.1, 0.1, 0.1]]
    metadatas = [{"title": "নমুনা", "url": "https://example.com", "source_id": "1"}]
    store.add_documents(ids=ids, texts=texts, embeddings=embeddings, metadatas=metadatas)
    docs = store.query([0.1, 0.1, 0.1], n_results=1)
    assert docs[0].page_content == "বাংলা চিকিৎসা"
