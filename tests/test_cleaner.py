from banglamed_rag.cleaner import chunk_sentences, clean_text


def test_clean_text_removes_english() -> None:
    text = (
        "This is a long English paragraph that should be removed by the cleaner. "
        "It contains enough words to trigger language detection.\n\n"
        "এটি একটি বাংলা অনুচ্ছেদ যা রাখা উচিত।"
    )
    cleaned = clean_text(text)
    assert "English" not in cleaned
    assert "বাংলা" in cleaned


def test_chunk_sentences_overlap() -> None:
    sentences = ["এক", "দুই", "তিন", "চার", "পাঁচ", "ছয়"]
    chunks = chunk_sentences(sentences, max_tokens=3, overlap=1)
    assert len(chunks) >= 2
    assert chunks[0].endswith("তিন")
    assert chunks[1].startswith("তিন")
