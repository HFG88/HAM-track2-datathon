"""Find the Hagstofa tables that best match a normalised question.

Two searches, which can be combined with reciprocal rank fusion (RRF).
Hybrid is the default. On AI_test/spurningar.txt the modes are close at #1
(BM25 33/83, hybrid 36/83, cascade 37/83), but hybrid most often has the right
table among its top 10-20 (55/83 and 60/83), which is what the LLM choice in
table_choice.py picks from. Cascade drops the right table on short questions.

- BM25 over character 4-grams, so different Icelandic word endings and
  compounds still match ("atvinnulausra" ~ "atvinnuleysi"). Each table is
  indexed in four fields with different weights: title, folder path, variable
  names and value labels.
- Vector search: each table's title, folders and variable names are embedded
  with a multilingual model (qwen3-embedding via Ollama), and compared with
  the embedded question. This catches meaning that shares no letters with
  the title. Vectors are cached in local/.

Modes: "bm25", "vector", "hybrid" (RRF of both) and "cascade" (vectors pick
the topic, then BM25 ranks the tables within it).

Input is the dict from normalize.normalize_question().
"""

import json
import os
import math
import re
import sys
from collections import Counter, defaultdict
from datetime import date
from functools import lru_cache
from pathlib import Path
from urllib.parse import unquote

import numpy as np
import requests

from normalize import QuestionError, clean, normalize_question, tokenize

LOCAL = Path(__file__).parent / "local"
CATALOG_PATHS = [LOCAL / "catalog-enriched.json", LOCAL / "catalog-full.json"]

# Where Ollama runs. Set OLLAMA_HOST to use another machine, e.g. through an SSH tunnel.
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
if "://" not in OLLAMA_HOST:  # the ollama CLI also accepts "host:port"
    OLLAMA_HOST = "http://" + OLLAMA_HOST
OLLAMA_EMBED_URL = OLLAMA_HOST + "/api/embed"
EMBED_MODEL = "qwen3-embedding:0.6b"
EMBED_BATCH = 32
# Qwen3 embeddings work best when the question says what it is looking for.
QUERY_INSTRUCTION = (
    "Instruct: Given a question about Icelandic official statistics, "
    "retrieve the title of the statistics table that answers it\nQuery: "
)
# What each table is represented by in vector search: "title" or "full"
# (title, folder names and variable names).
VECTOR_TEXT = "title"
# Cascade: how many vector results BM25 gets to rank.
CASCADE_CANDIDATES = 100
# Cascade: how many of BM25's own top tables are kept even if vectors missed them.
CASCADE_BM25_KEEP = 0

# RRF: each search adds 1 / (RRF_K + rank). Tables found by both rise to the top.
RRF_K = 60
CANDIDATES = 100
# How much vector search counts in hybrid compared to BM25 (1.0).
VECTOR_WEIGHT = 0.25

NGRAM = 4
K1 = 1.2
B = 0.75

# Tables whose newest year is older than this many years get a penalty.
OLD_AFTER_YEARS = 3
OLD_PENALTY = 0.3
# Historical tables (Sögulegar hagtölur) mostly duplicate current ones.
HISTORICAL_PENALTY = 0.2
SPECIFIC_PENALTY = 0.03

# How much a match in each field counts.
FIELD_WEIGHTS = {
    "title": 3.0,
    "path": 1.5,
    "variables": 1.5,
    "values": 0.5,
}


def ngrams(word):
    """Character n-grams of a word, padded so word starts and ends count too."""
    padded = f"_{word}_"
    if len(padded) <= NGRAM:
        return [padded]
    return [padded[i:i + NGRAM] for i in range(len(padded) - NGRAM + 1)]


def text_ngrams(text):
    grams = []
    for word in tokenize(clean(text)):
        if not word.isdigit():
            grams.extend(ngrams(word))
    return grams


def table_fields(table):
    """Split a catalog entry into the text of each indexed field."""
    metadata = table.get("metadata") or {}
    variables = metadata.get("variables", [])
    url_path = unquote(table["url"].split("/api/v1/")[-1])
    # Drop the language part ("is/") and the file name, keep the folders.
    folders = url_path.split("/")[1:-1]

    return {
        "title": table["title"],
        "path": " ".join(folders),
        "variables": " ".join(v.get("text", "") for v in variables),
        "values": " ".join(
            label
            for v in variables if not v.get("time")
            for label in v.get("valueTexts", [])
        ),
    }


class TableIndex:
    def __init__(self, tables):
        self.tables = tables
        self.postings = {}   # field -> ngram -> {doc: count}
        self.lengths = {}    # field -> [doc length]
        self.average = {}    # field -> average doc length

        for field in FIELD_WEIGHTS:
            postings = defaultdict(dict)
            lengths = []
            for doc, table in enumerate(tables):
                counts = Counter(text_ngrams(table_fields(table)[field]))
                lengths.append(sum(counts.values()))
                for gram, count in counts.items():
                    postings[gram][doc] = count
            self.postings[field] = postings
            self.lengths[field] = lengths
            self.average[field] = (sum(lengths) / len(lengths)) or 1

    def idf(self, field, gram):
        n = len(self.tables)
        df = len(self.postings[field].get(gram, ()))
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    def score_term(self, term, scores):
        """Add one query word's score to every table that matches it."""
        grams = set(ngrams(term))
        for field, weight in FIELD_WEIGHTS.items():
            postings = self.postings[field]
            lengths = self.lengths[field]
            average = self.average[field]
            for gram in grams:
                docs = postings.get(gram)
                if not docs:
                    continue
                idf = self.idf(field, gram)
                for doc, tf in docs.items():
                    norm = K1 * (1 - B + B * lengths[doc] / average)
                    bm25 = idf * tf * (K1 + 1) / (tf + norm)
                    # Divide by the number of n-grams so long words don't count more.
                    scores[doc] += weight * bm25 / len(grams)

    def search(self, normalized, limit=10):
        """Return [(table, score)] for the best matching tables."""
        scores = defaultdict(float)
        for term in query_terms(normalized):
            term_scores = defaultdict(float)
            self.score_term(term, term_scores)
            if not term_scores:
                continue
            # Scale each word to 0..1 so every word counts equally: a table must
            # match "mannfjoldi" and "akureyri", not just one of them strongly.
            best = max(term_scores.values())
            for doc, score in term_scores.items():
                scores[doc] += score / best

        for doc in scores:
            scores[doc] -= self.age_penalty(self.tables[doc])

        best = sorted(scores.items(), key=lambda item: -item[1])[:limit]
        return [(self.tables[doc], round(score, 3)) for doc, score in best]

    @staticmethod
    def age_penalty(table):
        penalty = 0.0
        if table.get("category") == "Sogulegar":
            penalty += HISTORICAL_PENALTY
        # Revision tables ("breytt aðferðafræði: Ofmat á fjölda íbúa") are not the official figures.
        if "/endurskodun/" in table["url"] or "ofmat" in clean(table["title"]):
            penalty += HISTORICAL_PENALTY
        latest = latest_year(table)
        if latest and latest < date.today().year - OLD_AFTER_YEARS:
            penalty += OLD_PENALTY
        # Prefer general tables: each variable beyond two makes a table more specific.
        variables = (table.get("metadata") or {}).get("variables", [])
        penalty += SPECIFIC_PENALTY * max(0, len(variables) - 2)
        return penalty


PERIOD = re.compile(r"^\d{4}(?:M\d{2}|\D?\d)?$")


TIME_CODES = {"Ár", "Mánuður", "Ársfjórðungur", "Tími", "Ár og mánuður", "Year", "Month", "Quarter"}


def is_time_variable(variable):
    """True for the time dimension. About half the tables lack the "time" flag, and
    many use codes 0, 1, 2 with the years only in the labels, so check codes and labels."""
    if variable.get("time") or variable.get("code") in TIME_CODES:
        return True
    # Judge by the labels: codes can be 4-digit numbers that are not years
    # (municipality 8722 = Flóahreppur).
    labels = variable.get("valueTexts") or variable.get("values", [])
    return bool(labels) and all(
        PERIOD.match(str(label).strip()) and 1700 <= int(str(label).strip()[:4]) <= 2100
        for label in labels
    )


def earliest_year(table):
    """Oldest year in the table, from the time variable if we have metadata."""
    for variable in (table.get("metadata") or {}).get("variables", []):
        if is_time_variable(variable) and variable.get("values"):
            for value in (variable["values"][0], (variable.get("valueTexts") or [""])[0]):
                match = re.search(r"(?:18|19|20)\d{2}", str(value))
                if match:
                    return int(match.group())
    years = [int(y) for y in re.findall(r"\b(?:18|19|20)\d{2}\b", table["title"])]
    return min(years, default=None)


def latest_year(table):
    """Newest year in the table, from the time variable if we have metadata."""
    for variable in (table.get("metadata") or {}).get("variables", []):
        if is_time_variable(variable) and variable.get("values"):
            for value in (variable["values"][-1], (variable.get("valueTexts") or [""])[-1]):
                match = re.search(r"(?:18|19|20)\d{2}", str(value))
                if match:
                    return int(match.group())
    years = [int(y) for y in re.findall(r"\b(?:18|19|20)\d{2}\b", table["title"])]
    return max(years, default=None)


def query_terms(normalized):
    """The words from a normalised question that are used for searching.

    ISKeywords (Icelandic keywords from the LLM rewrite, see question_router.py)
    are included when present, so English questions can match Icelandic titles.
    """
    if normalized.get("language") == "en":
        # English words don't occur in the Icelandic titles; rely on the translations.
        terms = list(normalized["expanded"])
    else:
        terms = normalized["keywords"] + normalized["expanded"] + normalized["breakdowns"]
    for keyword in normalized.get("ISKeywords", []):
        terms += tokenize(clean(keyword))
    return list(dict.fromkeys(t for t in terms if not t.isdigit()))


def embed(texts, model=None):
    """Embed a list of texts with Ollama. Returns an array of unit vectors."""
    vectors = []
    for i in range(0, len(texts), EMBED_BATCH):
        response = requests.post(
            OLLAMA_EMBED_URL,
            json={"model": model or EMBED_MODEL, "input": texts[i:i + EMBED_BATCH]},
            timeout=600,
        )
        response.raise_for_status()
        vectors.extend(response.json()["embeddings"])
    vectors = np.array(vectors, dtype=np.float32)
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def table_text(table, kind=None):
    """The text that represents a table in vector search."""
    if (kind or VECTOR_TEXT) == "title":
        return table["title"]
    fields = table_fields(table)
    return f"{table['title']}. {fields['path']}. {fields['variables']}"


def vectors_path(model, kind):
    safe = "".join(c if c.isalnum() else "-" for c in model)
    return LOCAL / f"table-vectors-{safe}-{kind}.npz"


class VectorIndex:
    def __init__(self, tables, model=None, kind=None):
        self.tables = tables
        self.model = model or EMBED_MODEL
        kind = kind or VECTOR_TEXT
        path = vectors_path(self.model, kind)
        urls = [t["url"] for t in tables]

        if path.exists():
            cached = np.load(path)
            if list(cached["urls"]) == urls:
                self.vectors = cached["vectors"]
                return

        print(f"Embedding {len(tables)} tables with {self.model} ({kind}, one-time)...", file=sys.stderr)
        self.vectors = embed([table_text(t, kind) for t in tables], self.model)
        np.savez(path, vectors=self.vectors, urls=np.array(urls))

    def search(self, question, limit=10):
        prefix = QUERY_INSTRUCTION if "qwen" in self.model else ""
        query = embed([prefix + question], self.model)[0]
        similarity = self.vectors @ query
        best = np.argsort(-similarity)[:limit]
        return [(self.tables[i], round(float(similarity[i]), 3)) for i in best]


@lru_cache(maxsize=1)
def load_tables():
    path = next((p for p in CATALOG_PATHS if p.exists()), None)
    if path is None:
        raise FileNotFoundError("No catalog in local/. Run catalog.py and enrich_catalog.py first.")
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def default_index():
    return TableIndex(load_tables())


@lru_cache(maxsize=4)
def default_vectors(model=None, kind=None):
    return VectorIndex(load_tables(), model or EMBED_MODEL, kind or VECTOR_TEXT)


def rrf(rankings, weights, limit=10):
    """Combine ranked lists of (table, score) by weighted reciprocal rank fusion."""
    scores = defaultdict(float)
    tables = {}
    for ranking, weight in zip(rankings, weights):
        for rank, (table, _) in enumerate(ranking, start=1):
            scores[table["url"]] += weight / (RRF_K + rank)
            tables[table["url"]] = table
    best = sorted(scores.items(), key=lambda item: -item[1])[:limit]
    return [(tables[url], round(score, 4)) for url, score in best]


def search_tables(normalized, limit=10, mode="hybrid"):
    """mode: "bm25", "vector", "hybrid" (RRF of both) or "cascade" (vectors, then BM25)."""
    if mode == "bm25":
        return default_index().search(normalized, limit)

    try:
        vector_results = default_vectors().search(normalized["original"], CANDIDATES)
    except requests.RequestException as error:
        # Ollama is down: fall back to BM25 alone.
        print(f"Vector search unavailable ({error}), using BM25 only.", file=sys.stderr)
        return default_index().search(normalized, limit)

    if mode == "vector":
        return vector_results[:limit]

    bm25_results = default_index().search(normalized, len(default_index().tables))
    if mode == "cascade":
        # Vectors decide the topic, BM25 decides the table within it.
        bm25_scores = {table["url"]: score for table, score in bm25_results}
        candidates = vector_results[:CASCADE_CANDIDATES]
        # Also keep BM25's own best tables: vectors miss e.g. which tables have
        # a municipality like "Kópavogur" among their values.
        seen = {table["url"] for table, _ in candidates}
        candidates += [item for item in bm25_results[:CASCADE_BM25_KEEP] if item[0]["url"] not in seen]
        ranked = sorted(
            candidates,
            key=lambda item: -bm25_scores.get(item[0]["url"], float("-inf")),
        )
        return [(table, bm25_scores.get(table["url"], 0.0)) for table, _ in ranked[:limit]]

    return rrf([bm25_results[:CANDIDATES], vector_results], [1.0, VECTOR_WEIGHT], limit=limit)


def describe_candidates(results, examples=4):
    """Short descriptions of search results, small enough for an LLM prompt.

    The url is the table's unique key: some ids exist in more than one folder.
    """
    candidates = []
    for table, score in results:
        variables = []
        period = None
        for variable in (table.get("metadata") or {}).get("variables", []):
            labels = variable.get("valueTexts", [])
            if is_time_variable(variable):
                period = f"{labels[0]} – {labels[-1]}" if labels else None
                continue
            variables.append({
                "name": variable.get("text", ""),
                "count": len(labels),
                "examples": labels[:examples],
            })
        candidates.append({
            "url": table["url"],
            "id": table["id"],
            "title": table["title"],
            "period": period,
            "variables": variables,
            "score": score,
        })
    return candidates


if __name__ == "__main__":
    args = sys.argv[1:]
    mode = "hybrid"
    if args and args[0] in ("--bm25", "--vector", "--hybrid", "--cascade"):
        mode = args.pop(0)[2:]
    questions = args or ["Hver er mannfjöldi í Kópavogi?"]
    for question in questions:
        try:
            normalized = normalize_question(question)
        except QuestionError as error:
            print(f"{question}\n  {error}")
            continue
        print(question)
        for table, score in search_tables(normalized, limit=5, mode=mode):
            print(f"  {score:7.3f}  {table['id']:14} {table['title']}")
