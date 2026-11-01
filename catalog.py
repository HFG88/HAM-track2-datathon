
import argparse
import time
import json
import re
import unicodedata
from pathlib import Path

from urllib.parse import quote

import requests

BASE_URLS = {
    "is": "https://px.hagstofa.is/pxis/api/v1/is/",
    "en": "https://px.hagstofa.is/pxen/api/v1/en/",
}
BASE_URL = BASE_URLS["is"]  # Keep existing imports/callers compatible.


def get_catalog_level(session, path, base_url=BASE_URL):
    """Retrieve one directory from the Statistics Iceland API."""
    url = base_url + "/".join(quote(p, safe="") for p in path)

    response = session.get(url, timeout=20)
    response.raise_for_status()

    entries = response.json()
    if not isinstance(entries, list):
        raise ValueError(f"Unexpected response from {url}")

    return entries


def crawl_tables(start=None, max_directories=None, language="is"):
    """Discover tables in Icelandic or English using the official PxWeb APIs."""
    base_url = BASE_URLS[language]
    stack = [start.strip("/").split("/")] if start else [[]]
    visited = set()
    tables = []
    category_labels = {}

    with requests.Session() as session:
        while stack and (max_directories is None or len(visited) < max_directories):
            path = stack.pop()
            path_string = "/".join(path)

            if path_string in visited:
                continue

            print(f"Scanning: {path_string}")
            entries = get_catalog_level(session, path, base_url=base_url)
            if not path:
                category_labels = {
                    entry.get("id") or entry.get("dbid"): entry.get("text", "")
                    for entry in entries
                }
            visited.add(path_string)

            for entry in reversed(entries):
                entry_id = entry.get("id") or entry.get("dbid")
                entry_type = entry.get("type")

                if not entry_id:
                    continue

                child_path = path + [entry_id]

                if entry_type == "t" or entry_id.endswith(".px"):
                    tables.append({
                        "id": entry_id,
                        "title": entry.get("text", ""),
                        "url": base_url + "/".join(
                            quote(p, safe="") for p in child_path
                        ),
                        "category": path[0] if path else entry_id,
                        "category_label": category_labels.get(path[0], path[0])
                        if path else entry.get("text", ""),
                    })

                elif entry_type in ("l", None):
                    stack.append(child_path)

                else:
                    print(f"Unknown entry type: {entry_type}")

            time.sleep(0.15)

    return tables


def normalize_words(text):
    """Normalize Icelandic text for basic keyword matching."""
    text = unicodedata.normalize("NFKD", text.casefold())
    text = "".join(
        c for c in text if not unicodedata.combining(c)
    )
    text = (
        text.replace("þ", "th")
            .replace("ð", "d")
            .replace("æ", "ae")
    )
    return set(re.findall(r"[a-z0-9]+", text))


def save_catalog(tables, filename="local/catalog-preview.json"):
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(
        json.dumps(tables, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    print(f"Saved {len(tables)} tables to {path}")



def search_catalog(tables, question, limit=10):
    """Rank tables by title relevance and available year hints."""
    query_words = {
        word for word in normalize_words(question)
        if len(word) >= 3
    }

    query_years = [
        int(y) for y in re.findall(r"\b(?:19|20)\d{2}\b", question)
    ]

    results = []

    for table in tables:
        title = table["title"]
        title_words = normalize_words(title)
        match_score = len(query_words & title_words)

        if match_score == 0:
            continue

        title_years = [
            int(y) for y in re.findall(
                r"\b(?:19|20)\d{2}\b", title
            )
        ]

        latest_year = max(title_years, default=0)
        covers_year = False

        if query_years and len(title_years) >= 2:
            covers_year = all(
                min(title_years) <= year <= max(title_years)
                for year in query_years
            )

        results.append((
            match_score,
            int(covers_year),
            latest_year,
            table
        ))

    results.sort(
        key=lambda item: (-item[0], -item[1], -item[2], item[3]["title"])
    )

    return [item[3] for item in results[:limit]]




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crawl Statistics Iceland's PxWeb table catalogue")
    parser.add_argument("--language", choices=("is", "en"), default="is")
    parser.add_argument("--start", default=None, help="Optional starting category/path")
    parser.add_argument("--max-directories", type=int, default=None, help="Limit folders for a test crawl")
    parser.add_argument("--output", default=None, help="Output JSON path")
    args = parser.parse_args()

    tables = crawl_tables(
        start=args.start,
        max_directories=args.max_directories,
        language=args.language,
    )

    default_output = (
        "local/catalog-full-en.json" if args.language == "en"
        else "local/catalog-full.json"
    )
    save_catalog(tables, args.output or default_output)

    categories = sorted(set(t["category"] for t in tables))
    print("\n=== CRAWL COMPLETE ===")
    print(f"Language: {args.language}")
    print(f"Total tables: {len(tables)}")
    print(f"Categories: {categories}")
