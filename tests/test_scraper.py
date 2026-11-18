from banglamed_rag.scraper import extract_text_from_html, save_raw_html


def test_extract_text_from_html() -> None:
    html = "<html><h1>শিরোনাম</h1><p>এটি বাংলা লেখা।</p></html>"
    text, title = extract_text_from_html("https://example.com", html)
    assert "বাংলা" in text
    assert "শিরোনাম" in title


def test_save_raw_html(tmp_path) -> None:
    docs = [{"id": "doc1", "html": "<html></html>"}]
    save_raw_html(docs, raw_dir=tmp_path)
    assert (tmp_path / "doc1.html").exists()
