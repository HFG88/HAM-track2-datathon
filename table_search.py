"""Find the Hagstofa tables that best match a normalised question.

BM25 over character 4-grams, so different Icelandic word endings and compounds
still match ("atvinnulausra" ~ "atvinnuleysi"). Each table is indexed in four
fields with different weights: title, folder path, variable names and value
labels. Input is the dict from normalize.normalize_question().
"""

import json
import math
import re
import sys
from collections import Counter, defaultdict
from datetime import date
from functools import lru_cache
from pathlib import Path
from urllib.parse import unquote

from normalize import QuestionError, clean, normalize_question, tokenize

LOCAL = Path(__file__).parent / "local"
CATALOG_PATHS = [LOCAL / "catalog-enriched.json", LOCAL / "catalog-full.json"]

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
        latest = latest_year(table)
        if latest and latest < date.today().year - OLD_AFTER_YEARS:
            penalty += OLD_PENALTY
        # Prefer general tables: each variable beyond two makes a table more specific.
        variables = (table.get("metadata") or {}).get("variables", [])
        penalty += SPECIFIC_PENALTY * max(0, len(variables) - 2)
        return penalty


def latest_year(table):
    """Newest year in the table, from the time variable if we have metadata."""
    for variable in (table.get("metadata") or {}).get("variables", []):
        if variable.get("time") and variable.get("values"):
            match = re.match(r"\d{4}", variable["values"][-1])
            if match:
                return int(match.group())
    years = [int(y) for y in re.findall(r"\b(?:18|19|20)\d{2}\b", table["title"])]
    return max(years, default=None)


def query_terms(normalized):
    """The words from a normalised question that are used for searching."""
    terms = normalized["keywords"] + normalized["expanded"] + normalized["breakdowns"]
    return list(dict.fromkeys(t for t in terms if not t.isdigit()))


@lru_cache(maxsize=1)
def default_index():
    path = next((p for p in CATALOG_PATHS if p.exists()), None)
    if path is None:
        raise FileNotFoundError("No catalog in local/. Run catalog.py and enrich_catalog.py first.")
    return TableIndex(json.loads(path.read_text(encoding="utf-8")))


def search_tables(normalized, limit=10):
    return default_index().search(normalized, limit)


if __name__ == "__main__":
    questions = sys.argv[1:] or ["Hver er mannfjöldi í Kópavogi?"]
    for question in questions:
        try:
            normalized = normalize_question(question)
        except QuestionError as error:
            print(f"{question}\n  {error}")
            continue
        print(question)
        for table, score in search_tables(normalized, limit=5):
            print(f"  {score:7.2f}  {table['id']:14} {table['title']}")
