"""Detect a question's language and route it to the matching rewrite pipeline."""

import json
import sys

from llm_rewrite_is import MODEL, rewrite as rewrite_icelandic
from llm_rewrite_en import rewrite as rewrite_english
from normalize import clean, normalize_question

ICELANDIC_MARKERS = {
    "hvad", "hvada", "hver", "hverjir", "hverjar", "hversu", "hvernig", "hvar",
    "hve", "margir", "margar", "morg", "mikid", "eru", "var", "voru", "fra",
    "fjoldi", "arid", "sveitarfelag", "mannfjoldi", "verd", "verdbolga",
    "atvinnuleysi", "laun",
}
ENGLISH_MARKERS = {
    "what", "which", "who", "where", "when", "how", "many", "much", "people",
    "are", "was", "were", "does", "did", "is", "unemployment", "population",
    "inflation", "wages", "salary", "tourists", "tourism", "births", "migration",
}


def detect_language(question):
    """Return 'is' or 'en' using common question words and topic vocabulary."""
    words = set(clean(question).split())
    icelandic_score = len(words & ICELANDIC_MARKERS)
    english_score = len(words & ENGLISH_MARKERS)

    if any(character in question.lower() for character in "áðéíóúýþæö"):
        icelandic_score += 2

    return "en" if english_score > icelandic_score else "is"


def rewrite_question(question, model=MODEL):
    """Return normalized question data with language-specific LLM keywords."""
    language = detect_language(question)
    normalized = normalize_question(question)

    if language == "en":
        rewritten = rewrite_english(question, model=model)
    else:
        rewritten = rewrite_icelandic(normalized, model=model)

    normalized["ISKeywords"] = rewritten.get("ISKeywords", [])
    normalized["ENKeywords"] = rewritten.get("ENKeywords", [])
    if "llm_error" in rewritten:
        normalized["llm_error"] = rewritten["llm_error"]
    normalized["language"] = language
    return normalized


if __name__ == "__main__":
    questions = sys.argv[1:] or [
        "Hvað eru margir án vinnu?",
        "How many people are unemployed?",
    ]

    for question in questions:
        try:
            result = rewrite_question(question)
        except ValueError as error:
            print(f"{question}\n  {error}")
            continue

        print(json.dumps(result, ensure_ascii=False, indent=2))
