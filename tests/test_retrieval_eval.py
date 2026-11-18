from banglamed_rag.retrieval_eval import bm25_ranker, embedding_ranker, score_rankings, tokenize


def test_tokenize_keeps_bengali_words() -> None:
    assert tokenize("ডেঙ্গুর লক্ষণ, কী?") == ["ডেঙ্গুর", "লক্ষণ", "কী"]


def test_score_rankings() -> None:
    rankings = [[0, 1, 2], [0, 1, 2], [0, 1, 2]]
    metrics = score_rankings(rankings, ks=(1, 3))
    assert metrics["hit@1"] == 1 / 3
    assert metrics["hit@3"] == 1.0
    assert abs(metrics["mrr"] - (1 + 1 / 2 + 1 / 3) / 3) < 1e-9


def test_bm25_ranks_matching_passage_first() -> None:
    passages = ["ডেঙ্গুতে উচ্চ জ্বর হয়", "ডায়াবেটিসে গ্লুকোজ বাড়ে"]
    rank = bm25_ranker(passages)
    assert rank("ডায়াবেটিসে কী হয়")[0] == 1
    assert rank("ডেঙ্গুতে জ্বর")[0] == 0


def test_embedding_ranker_uses_cosine_similarity() -> None:
    vectors = {"a": [1.0, 0.0], "b": [0.0, 1.0], "qb": [0.1, 0.9]}

    def embed(texts):
        return [vectors[t] for t in texts]

    rank = embedding_ranker(["a", "b"], embed)
    assert rank("qb") == [1, 0]
