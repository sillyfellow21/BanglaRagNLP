from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

from banglamed_rag.rag_chain import BanglaMedRAG


class DummyVectorStore:
    def query(self, query_embedding, n_results: int = 5):
        return [
            Document(
                page_content="এই অংশে তথ্য রয়েছে।",
                metadata={"title": "শিরোনাম", "url": "u", "source_id": "1"},
            )
        ]


class DummyEmbedder:
    def embed_documents(self, texts):
        return [[0.0] * 3 for _ in texts]

    def embed_query(self, text):
        return [0.0] * 3


def test_rag_chain_assembly() -> None:
    llm = RunnableLambda(lambda _: "উত্তর")
    rag = BanglaMedRAG(vectorstore=DummyVectorStore(), embedder=DummyEmbedder(), llm=llm)
    result = rag.ask("প্রশ্ন", top_k=1)
    assert "উত্তর" in result["answer"]
    assert result["sources"]
