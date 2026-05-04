from __future__ import annotations

from typing import Any

from ragas import evaluate
from ragas.embeddings import LangchainEmbeddings
from ragas.llms import LangchainLLM
from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import HuggingFaceEmbeddings

from .config import settings
from .rag_chain import BanglaMedRAG
from .utils import load_json, save_json

try:
    from datasets import Dataset
except Exception:  # pragma: no cover
    Dataset = None


def build_dataset(samples: list[dict[str, Any]]) -> Any:
    if Dataset is None:
        raise RuntimeError("datasets is required for evaluation")
    return Dataset.from_list(samples)


def run_evaluation(top_k: int | None = None) -> dict[str, Any]:
    qa_data = load_json(settings.qa_benchmark_path)
    rag = BanglaMedRAG()
    samples: list[dict[str, Any]] = []
    for item in qa_data:
        question = item["question"]
        ground_truth = item["ground_truth"]
        retrieval = rag._retrieve(question, top_k or settings.default_top_k)
        docs = rag._to_documents(retrieval)
        response = rag.ask(question, top_k=top_k)
        contexts = [doc.page_content for doc in docs]
        samples.append(
            {
                "question": question,
                "answer": response["answer"],
                "contexts": contexts,
                "ground_truth": ground_truth,
            }
        )
    dataset = build_dataset(samples)
    llm = LangchainLLM(
        ChatOllama(model=settings.ollama_model, base_url=settings.ollama_base_url)
    )
    embeddings = LangchainEmbeddings(HuggingFaceEmbeddings(model_name=settings.embed_model))
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=llm,
        embeddings=embeddings,
    )
    output: dict[str, Any]
    try:
        scores = result.to_pandas().mean(numeric_only=True).to_dict()
        per_sample = result.to_pandas().to_dict(orient="records")
        output = {"summary": scores, "per_sample": per_sample}
    except Exception:
        output = {"summary": getattr(result, "scores", {})}
    save_json(settings.evaluation_results_path, output)
    return output


def main() -> None:
    results = run_evaluation()
    print("Evaluation summary:")
    for key, value in results.get("summary", {}).items():
        print(f"- {key}: {value}")


if __name__ == "__main__":
    main()
