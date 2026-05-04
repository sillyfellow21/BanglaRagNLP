from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI
from pydantic import BaseModel, Field

from banglamed_rag.rag_chain import BanglaMedRAG

app = FastAPI(title="BanglaMed-RAG")


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3)
    top_k: int | None = Field(default=None, ge=1, le=20)


class Source(BaseModel):
    id: str
    title: str
    snippet: str


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: list[Source]


@lru_cache
def get_rag() -> BanglaMedRAG:
    return BanglaMedRAG()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(payload: AskRequest) -> AskResponse:
    rag = get_rag()
    result = rag.ask(payload.question, top_k=payload.top_k)
    return AskResponse(**result)
