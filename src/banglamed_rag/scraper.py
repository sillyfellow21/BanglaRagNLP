from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import requests
from bs4 import BeautifulSoup

try:
    from newspaper import Article
except Exception:  # pragma: no cover
    Article = None

from .config import settings
from .utils import ensure_dir, normalize_whitespace, slugify, write_jsonl

WIKI_API = "https://bn.wikipedia.org/w/api.php"
CATEGORIES = ["স্বাস্থ্য", "চিকিৎসা", "রোগ", "পুষ্টি", "মাতৃত্ব"]

FALLBACK_URLS = [
    "https://bn.wikipedia.org/wiki/ডেঙ্গু",
    "https://bn.wikipedia.org/wiki/ডায়াবেটিস",
    "https://bn.wikipedia.org/wiki/জ্বর",
    "https://bn.wikipedia.org/wiki/কলেরা",
    "https://bn.wikipedia.org/wiki/যক্ষ্মা",
    "https://bn.wikipedia.org/wiki/ম্যালেরিয়া",
    "https://bn.wikipedia.org/wiki/হেপাটাইটিস",
    "https://bn.wikipedia.org/wiki/এইডস",
    "https://bn.wikipedia.org/wiki/অ্যানিমিয়া",
    "https://bn.wikipedia.org/wiki/গর্ভাবস্থা",
    "https://bn.wikipedia.org/wiki/প্রসব",
    "https://bn.wikipedia.org/wiki/স্তন্যপান",
    "https://bn.wikipedia.org/wiki/পুষ্টি",
    "https://bn.wikipedia.org/wiki/ভিটামিন",
    "https://bn.wikipedia.org/wiki/রক্তচাপ",
    "https://bn.wikipedia.org/wiki/হৃদরোগ",
    "https://bn.wikipedia.org/wiki/কিডনি",
    "https://bn.wikipedia.org/wiki/ক্যান্সার",
    "https://bn.wikipedia.org/wiki/অ্যাজমা",
    "https://bn.wikipedia.org/wiki/অ্যালার্জি",
    "https://bn.wikipedia.org/wiki/ডায়রিয়া",
    "https://bn.wikipedia.org/wiki/সর্দি",
    "https://bn.wikipedia.org/wiki/ইনফ্লুয়েঞ্জা",
    "https://bn.wikipedia.org/wiki/হেপাটাইটিস_বি",
    "https://bn.wikipedia.org/wiki/টিকা",
]


def fetch_page_html(url: str, session: requests.Session) -> str:
    if not url:
        return ""
    try:
        response = session.get(url, timeout=30)
        response.raise_for_status()
        return response.text
    except requests.RequestException:
        return ""


def extract_text_from_html(url: str, html: str) -> tuple[str, str]:
    title = ""
    text = ""
    if Article is not None and html:
        article = Article(url=url, language="bn")
        article.set_html(html)
        article.parse()
        text = article.text or ""
        title = article.title or ""
    if html:
        soup = BeautifulSoup(html, "html.parser")
        title_el = soup.find("h1")
        if not title:
            title = title_el.get_text(strip=True) if title_el else ""
        if not text:
            content = soup.select_one("#mw-content-text")
            text = content.get_text(separator=" ") if content else soup.get_text(" ")
    return normalize_whitespace(text), title


def save_raw_html(documents: list[dict[str, Any]], raw_dir: Path | None = None) -> None:
    raw_dir = raw_dir or settings.raw_dir
    ensure_dir(raw_dir)
    for doc in documents:
        html = doc.get("html", "")
        if not html:
            continue
        raw_path = raw_dir / f"{doc['id']}.html"
        raw_path.write_text(html, encoding="utf-8")


def prepare_corpus_records(documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for doc in documents:
        records.append(
            {
                "id": doc.get("id", ""),
                "title": doc.get("title", ""),
                "text": doc.get("text", ""),
                "url": doc.get("url", ""),
                "source": doc.get("source", ""),
            }
        )
    return records


def fetch_category_titles(category: str, max_pages: int) -> list[str]:
    titles: list[str] = []
    params: dict[str, Any] = {
        "action": "query",
        "list": "categorymembers",
        "cmtitle": f"Category:{category}",
        "cmlimit": 500,
        "format": "json",
    }
    session = requests.Session()
    session.headers.update({"User-Agent": settings.user_agent})
    while True:
        response = session.get(WIKI_API, params=params, timeout=30)
        response.raise_for_status()
        payload = response.json()
        members = payload.get("query", {}).get("categorymembers", [])
        for member in members:
            title = member.get("title")
            if title:
                titles.append(title)
        if "continue" not in payload or len(titles) >= max_pages:
            break
        params["cmcontinue"] = payload["continue"]["cmcontinue"]
        time.sleep(0.2)
    return titles[:max_pages]


def fetch_page_extract(title: str, session: requests.Session) -> dict[str, Any] | None:
    params = {
        "action": "query",
        "prop": "extracts|info",
        "explaintext": 1,
        "inprop": "url",
        "titles": title,
        "format": "json",
    }
    response = session.get(WIKI_API, params=params, timeout=30)
    response.raise_for_status()
    pages = response.json().get("query", {}).get("pages", {})
    page = next(iter(pages.values()), None)
    if not page or "extract" not in page:
        return None
    url = page.get("fullurl", "")
    html = fetch_page_html(url, session)
    return {
        "id": slugify(title),
        "title": page.get("title", title),
        "url": url,
        "text": normalize_whitespace(page.get("extract", "")),
        "html": html,
        "source": "wikipedia",
    }


def scrape_wikipedia(max_pages: int = 500) -> list[dict[str, Any]]:
    titles: list[str] = []
    for category in CATEGORIES:
        titles.extend(fetch_category_titles(category, max_pages=max_pages))
        if len(titles) >= max_pages:
            break
    unique_titles = list(dict.fromkeys(titles))[:max_pages]
    documents: list[dict[str, Any]] = []
    session = requests.Session()
    session.headers.update({"User-Agent": settings.user_agent})
    for title in unique_titles:
        try:
            doc = fetch_page_extract(title, session)
        except requests.RequestException:
            continue
        if doc and doc.get("text"):
            documents.append(doc)
    return documents


def scrape_fallback_urls() -> list[dict[str, Any]]:
    documents: list[dict[str, Any]] = []
    session = requests.Session()
    session.headers.update({"User-Agent": settings.user_agent})
    for url in FALLBACK_URLS:
        try:
            response = session.get(url, timeout=30)
            response.raise_for_status()
        except requests.RequestException:
            continue
        html = response.text
        text, title_text = extract_text_from_html(url, html)
        if not title_text:
            title_text = url.rsplit("/", 1)[-1]
        documents.append(
            {
                "id": slugify(title_text),
                "title": title_text,
                "url": url,
                "text": text,
                "html": html,
                "source": "fallback",
            }
        )
    return documents


def run_scraper(max_pages: int = 500, min_docs: int = 25) -> list[dict[str, Any]]:
    ensure_dir(settings.processed_dir)
    documents = []
    try:
        documents = scrape_wikipedia(max_pages=max_pages)
    except requests.RequestException:
        documents = []
    if len(documents) < min_docs:
        documents = scrape_fallback_urls()
    save_raw_html(documents)
    write_jsonl(settings.corpus_path, prepare_corpus_records(documents))
    return documents


def main() -> None:
    docs = run_scraper()
    print(f"Saved {len(docs)} documents to {settings.corpus_path}")


if __name__ == "__main__":
    main()
