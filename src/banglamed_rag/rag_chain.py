from __future__ import annotations

from typing import Any

from langchain_community.llms import Ollama
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from .config import settings
from .embedder import BanglaEmbedder
from .vectorstore import ChromaVectorStore


class BanglaMedRAG:
    def __init__(
        self,
        embedder: BanglaEmbedder | None = None,
        vectorstore: ChromaVectorStore | None = None,
        llm: Any | None = None,
        llm_model_name: str | None = None,
        temperature: float = 0.2,
        default_top_k: int | None = None,
    ) -> None:
        self.embedder = embedder or BanglaEmbedder()
        self.vectorstore = vectorstore or ChromaVectorStore(self.embedder)
        self.default_top_k = default_top_k or settings.default_top_k
        model_name = llm_model_name or settings.ollama_model
        self.llm = llm or Ollama(
            model=model_name,
            base_url=settings.ollama_base_url,
            temperature=temperature,
        )
        self.prompt = PromptTemplate.from_template(
            "তুমি একজন বাংলা চিকিৎসা সহকারী। কেবল প্রদত্ত প্রসঙ্গ ব্যবহার করে উত্তর দাও। "
            "প্রসঙ্গে তথ্য না থাকলে বলো: 'দুঃখিত, প্রদত্ত প্রসঙ্গে উত্তর নেই।' "
            "উত্তর সংক্ষিপ্ত ও স্পষ্ট বাংলায় দাও।\n\n"
            "প্রশ্ন: {question}\n\nপ্রসঙ্গ:\n{context}\n\nউত্তর:"
        )

    def _format_docs(self, documents: list[Document]) -> str:
        formatted: list[str] = []
        for doc in documents:
            title = doc.metadata.get("title", "")
            url = doc.metadata.get("url", "")
            snippet = doc.page_content
            formatted.append(f"শিরোনাম: {title}\nউৎস: {url}\nতথ্য: {snippet}")
        return "\n\n".join(formatted)

    def _retrieve(self, question: str, top_k: int) -> list[Document]:
        query_embedding = self.embedder.embed_query(question)
        return self.vectorstore.query(query_embedding, n_results=top_k)

    def _build_chain(self, top_k: int):
        return (
            {
                "context": RunnableLambda(
                    lambda question: self._format_docs(
                        self._retrieve(question, top_k)
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
        docs = self._retrieve(question, top_k)
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
