from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    project_root: Path = Path(__file__).resolve().parents[2]
    data_dir: Path = project_root / "data"
    raw_dir: Path = data_dir / "raw"
    processed_dir: Path = data_dir / "processed"
    corpus_path: Path = processed_dir / "corpus.jsonl"
    cleaned_corpus_path: Path = processed_dir / "corpus_clean.jsonl"
    qa_benchmark_path: Path = data_dir / "qa_benchmark.json"
    evaluation_results_path: Path = data_dir / "evaluation_results.json"
    chroma_path: Path = project_root / "chroma_db"
    collection_name: str = "banglamed_rag"
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "phi3")
    embed_model: str = os.getenv("EMBED_MODEL", "sagorsarker/bangla-bert-base")
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "512"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "50"))
    default_top_k: int = int(os.getenv("TOP_K", "5"))
    user_agent: str = os.getenv(
        "SCRAPER_USER_AGENT",
        "BanglaMed-RAG/1.0 (+https://github.com/sillyfellow21/BanglaRagNLP)",
    )


settings = Settings()
