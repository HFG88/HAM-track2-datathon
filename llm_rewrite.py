"""Ask a local LLM (Ollama) for English search words for a normalised question.

Llama is much better at English than Icelandic, so the LLM returns English
keywords that can be searched against the English table titles from
https://px.hagstofa.is/pxen/api/v1/en/ (same table ids as the Icelandic API).

If the English catalog exists, words that appear in no English title are
dropped, so the LLM cannot make up terms. If Ollama is unavailable the
question is returned unchanged.
"""

import json
import re
import sys
from functools import lru_cache
from pathlib import Path

import requests

from normalize import QuestionError, normalize_question

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "llama3.2:3b"  # try "llama3.1:8b" if you have the disk space
MAX_WORDS = 8
CATALOG_EN_PATH = Path(__file__).parent / "local" / "catalog-en.json"

SYSTEM_PROMPT = """The user asks an Icelandic question about official statistics.
Return 2-5 English keywords that would appear in the title of the Statistics Iceland table that answers it.
Use standard statistical terms (e.g. population, consumer price index, migration, unemployment, wages, births).
Do not include years, place names or question words. Answer only with JSON: {"keywords": [...]}"""

# Example question/answer pairs shown to the model before the real question.
EXAMPLES = [
    ("Hvað búa margir í Hafnarfirði?", ["population", "municipality"]),
    ("Hver var verðbólgan í fyrra?", ["consumer price index", "inflation"]),
]


@lru_cache(maxsize=1)
def english_vocabulary():
    """Words in the English table titles, or None if the catalog is not crawled yet."""
    if not CATALOG_EN_PATH.exists():
        return None
    tables = json.loads(CATALOG_EN_PATH.read_text(encoding="utf-8"))
    return frozenset(w for t in tables for w in re.findall(r"[a-z]+", t["title"].lower()))


def ask_llm(question, model=MODEL, timeout=30):
    """Return the raw list of keywords suggested by the LLM."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for example_question, keywords in EXAMPLES:
        messages.append({"role": "user", "content": example_question})
        messages.append({"role": "assistant", "content": json.dumps({"keywords": keywords})})
    messages.append({"role": "user", "content": question})

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": model,
            "messages": messages,
            "format": "json",
            "stream": False,
            # num_predict stops the model if it starts repeating itself.
            "options": {"temperature": 0, "num_predict": 60},
        },
        timeout=timeout,
    )
    response.raise_for_status()
    content = json.loads(response.json()["message"]["content"])
    return [k for k in content.get("keywords", []) if isinstance(k, str)]


def rewrite(normalized, model=MODEL):
    """Add an "llm_keywords_en" field with English search words."""
    normalized = dict(normalized)
    normalized["llm_keywords_en"] = []

    try:
        suggestions = ask_llm(normalized["original"], model=model)
    except (requests.RequestException, ValueError, KeyError) as error:
        normalized["llm_error"] = str(error)
        return normalized

    vocabulary = english_vocabulary()
    for suggestion in suggestions:
        for word in re.findall(r"[a-z]+", suggestion.lower()):
            if len(word) < 3 or word in normalized["llm_keywords_en"]:
                continue
            if vocabulary is None or word in vocabulary:
                normalized["llm_keywords_en"].append(word)

    normalized["llm_keywords_en"] = normalized["llm_keywords_en"][:MAX_WORDS]
    return normalized


if __name__ == "__main__":
    questions = sys.argv[1:] or [
        "Hvað eru margir án vinnu?",
        "Hvað kostar að leigja íbúð?",
        "Hvað komu margir ferðamenn til landsins 2024?",
        "Hver eru meðallaun kennara?",
        "Hversu mikið rafmagn er framleitt?",
        "Hversu margir fluttu frá Austurlandi 2023?",
        "Hvað fæddust mörg börn 2020?",
    ]
    for question in questions:
        try:
            result = rewrite(normalize_question(question))
        except QuestionError as error:
            print(f"{question}\n  {error}")
            continue
        print(question)
        print("  keywords:       ", result["keywords"] + result["expanded"])
        print("  llm_keywords_en:", result["llm_keywords_en"], result.get("llm_error", ""))
