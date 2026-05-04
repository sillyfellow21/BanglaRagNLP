from __future__ import annotations

from typing import Any

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_community.chat_models import ChatOllama

from .config import settings
from .embedder import BanglaEmbedder
from .vectorstore import ChromaVectorStore, QueryResult


class BanglaMedRAG:
    def __init__(
        self,
        embedder: BanglaEmbedder | None = None,
        vectorstore: ChromaVectorStore | None = None,
        llm: Any | None = None,
        temperature: float = 0.2,
        default_top_k: int | None = None,
    ) -> None:
        self.embedder = embedder or BanglaEmbedder()
        self.vectorstore = vectorstore or ChromaVectorStore(self.embedder)
        self.default_top_k = default_top_k or settings.default_top_k
        self.llm = llm or ChatOllama(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            temperature=temperature,
        )
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "তুমি একজন বাংলা চিকিৎসা সহকারী। কেবল প্রদত্ত প্রসঙ্গ ব্যবহার করে উত্তর দাও। "
                    "প্রসঙ্গে তথ্য না থাকলে বলো: 'দুঃখিত, প্রদত্ত প্রসঙ্গে উত্তর নেই।' "
                    "উত্তর সংক্ষিপ্ত ও স্পষ্ট বাংলায় দাও।",
                ),
                (
                    "human",
                    "প্রশ্ন: {question}\n\nপ্রসঙ্গ:\n{context}\n\nউত্তর:",
                ),
            ]
        )

    def _to_documents(self, result: QueryResult) -> list[Document]:
        documents: list[Document] = []
        for docs, metas in zip(result.documents, result.metadatas):
            for doc, meta in zip(docs, metas):
                documents.append(Document(page_content=doc, metadata=meta))
        return documents

    def _format_docs(self, documents: list[Document]) -> str:
        formatted: list[str] = []
        for doc in documents:
            title = doc.metadata.get("title", "")
            url = doc.metadata.get("url", "")
            snippet = doc.page_content
            formatted.append(f"শিরোনাম: {title}\nউৎস: {url}\nতথ্য: {snippet}")
        return "\n\n".join(formatted)

    def _retrieve(self, question: str, top_k: int) -> QueryResult:
        return self.vectorstore.query(question, n_results=top_k)

    def _build_chain(self, top_k: int):
        return (
            {
                "context": RunnableLambda(
                    lambda question: self._format_docs(
                        self._to_documents(self._retrieve(question, top_k))
                    )
                ),
                "question": RunnablePassthrough(),
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

    def ask(self, question: str, top_k: int | None = None) -> dict[str, Any]:
        top_k = top_k or self.default_top_k
        result = self._retrieve(question, top_k)
        docs = self._to_documents(result)
        chain = self._build_chain(top_k)
        answer = chain.invoke(question)
        sources = [
            {
                "id": doc.metadata.get("source_id", ""),
                "title": doc.metadata.get("title", ""),
                "snippet": doc.page_content[:240],
            }
            for doc in docs
        ]
        return {"question": question, "answer": answer, "sources": sources}
