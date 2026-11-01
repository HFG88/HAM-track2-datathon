"""Ask a local LLM (Ollama) for Icelandic and English search keywords.

Input:
    An Icelandic question about official statistics.

Output:
    Two separate keyword arrays:
    - ISKeywords: Icelandic keywords
    - ENKeywords: English translations, in the same order

Example:
    Input:
        "Hvað eru margir án vinnu?"

    Output:
        {
            "ISKeywords": ["vinnuleysi", "atvinnuleitendur"],
            "ENKeywords": ["unemployment", "jobseekers"]
        }

The English keywords can be searched against English Statistics Iceland
table titles, while the Icelandic keywords can be searched against
Icelandic table titles.
"""

import json
import os
import re
import sys
from functools import lru_cache
from pathlib import Path

import requests

from normalize import QuestionError, normalize_question

# Where Ollama runs. Set OLLAMA_HOST to use another machine, e.g. through an SSH tunnel.
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
if "://" not in OLLAMA_HOST:  # the ollama CLI also accepts "host:port"
    OLLAMA_HOST = "http://" + OLLAMA_HOST
OLLAMA_URL = OLLAMA_HOST + "/api/chat"
MODEL = "qwen3:8b"
MAX_KEYWORDS = 5

CATALOG_IS_PATH = Path(__file__).parent / "local" / "catalog-is.json"
CATALOG_EN_PATH = Path(__file__).parent / "local" / "catalog-en.json"

SYSTEM_PROMPT = """The user asks an Icelandic question about official statistics.
Return 2-5 keyword pairs for concepts that would appear in the title of the
Statistics Iceland table that answers the question.

Return exactly this JSON structure:
{
  "ISKeywords": ["Icelandic keyword 1", "Icelandic keyword 2"],
  "ENKeywords": ["English keyword 1", "English keyword 2"]
}

Rules:
- The arrays must have the same number of items.
- ENKeywords[i] must be the English translation of ISKeywords[i].
- Use standard statistical terms.
- Do not include years, place names, or question words.
- Answer only with JSON.
"""

EXAMPLES = [
    (
        "Hvað búa margir í Hafnarfirði?",
        {
            "ISKeywords": ["íbúafjöldi", "sveitarfélag"],
            "ENKeywords": ["population", "municipality"],
        },
    ),
    (
        "Hver var verðbólgan í fyrra?",
        {
            "ISKeywords": ["verðbólga", "neysluverðsvísitala"],
            "ENKeywords": ["inflation", "consumer price index"],
        },
    ),
    (
        "Hvað eru margir án vinnu?",
        {
            "ISKeywords": ["vinnuleysi", "atvinnuleitendur"],
            "ENKeywords": ["unemployment", "jobseekers"],
        },
    ),
]


@lru_cache(maxsize=1)
def title_vocabulary(path):
    """Return words used in table titles, or None if the catalog does not exist."""
    if not path.exists():
        return None

    tables = json.loads(path.read_text(encoding="utf-8"))
    words = set()

    for table in tables:
        words.update(
            re.findall(r"[a-záðéíóúýþæö]+", table["title"].lower())
        )

    return frozenset(words)


def clean_keyword(keyword, vocabulary=None):
    """Clean one keyword and optionally keep only words found in table titles."""
    if not isinstance(keyword, str):
        return []

    words = re.findall(r"[a-záðéíóúýþæö]+", keyword.lower())

    return [
        word
        for word in words
        if len(word) >= 3 and (vocabulary is None or word in vocabulary)
    ]


def ask_llm(question, model=MODEL, timeout=30):
    """Ask Ollama for separate Icelandic and English keyword arrays."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    for example_question, keywords in EXAMPLES:
        messages.append({"role": "user", "content": example_question})
        messages.append(
            {"role": "assistant", "content": json.dumps(keywords, ensure_ascii=False)}
        )

    messages.append({"role": "user", "content": question})

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": model,
            "messages": messages,
            "format": "json",
            "stream": False,
            # Qwen 3 can "think" before answering, which is slower.
            "think": False,
            "options": {
                "temperature": 0,
                "num_predict": 150,
            },
        },
        timeout=timeout,
    )

    response.raise_for_status()

    content = response.json()["message"]["content"]
    return json.loads(content)


def rewrite(normalized, model=MODEL):
    """Add separate Icelandic and English keyword arrays to a normalized question."""
    normalized = dict(normalized)

    normalized["ISKeywords"] = []
    normalized["ENKeywords"] = []

    try:
        suggestions = ask_llm(normalized["original"], model=model)
    except (requests.RequestException, ValueError, KeyError) as error:
        normalized["llm_error"] = str(error)
        return normalized

    icelandic_vocab = title_vocabulary(CATALOG_IS_PATH)
    english_vocab = title_vocabulary(CATALOG_EN_PATH)

    is_suggestions = suggestions.get("ISKeywords", [])
    en_suggestions = suggestions.get("ENKeywords", [])

    # Keep the lists aligned: only use complete Icelandic/English pairs.
    pair_count = min(len(is_suggestions), len(en_suggestions))

    for index in range(pair_count):
        is_words = clean_keyword(is_suggestions[index], icelandic_vocab)
        en_words = clean_keyword(en_suggestions[index], english_vocab)

        if not is_words or not en_words:
            continue

        is_keyword = " ".join(is_words)
        en_keyword = " ".join(en_words)

        if is_keyword in normalized["ISKeywords"]:
            continue

        if en_keyword in normalized["ENKeywords"]:
            continue

        normalized["ISKeywords"].append(is_keyword)
        normalized["ENKeywords"].append(en_keyword)

    normalized["ISKeywords"] = normalized["ISKeywords"][:MAX_KEYWORDS]
    normalized["ENKeywords"] = normalized["ENKeywords"][:MAX_KEYWORDS]

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
        print("  Original keywords:", result["keywords"] + result["expanded"])
        print("  ISKeywords:", result["ISKeywords"], result.get("llm_error", ""))
        print("  ENKeywords:", result["ENKeywords"])