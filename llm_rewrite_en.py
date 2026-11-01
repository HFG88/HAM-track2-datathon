"""Rewrite English statistics questions into Icelandic and English keywords."""

import json
import sys

import requests

from llm_rewrite_is import (
    CATALOG_EN_PATH,
    CATALOG_IS_PATH,
    MAX_KEYWORDS,
    MODEL,
    OLLAMA_URL,
    clean_keyword,
    title_vocabulary,
)

SYSTEM_PROMPT = """The user asks an English question about official statistics.
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
- Return only JSON.
"""

EXAMPLES = [
    (
        "How many people live in Hafnarfjordur?",
        {
            "ISKeywords": ["íbúafjöldi", "sveitarfélag"],
            "ENKeywords": ["population", "municipality"],
        },
    ),
    (
        "What was inflation last year?",
        {
            "ISKeywords": ["verðbólga", "neysluverðsvísitala"],
            "ENKeywords": ["inflation", "consumer price index"],
        },
    ),
    (
        "How many people are unemployed?",
        {
            "ISKeywords": ["vinnuleysi", "atvinnuleitendur"],
            "ENKeywords": ["unemployment", "jobseekers"],
        },
    ),
]


def ask_llm(question, model=MODEL, timeout=30):
    """Ask Ollama for Icelandic and English table-search keyword pairs."""
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
            "think": False,
            "options": {"temperature": 0, "num_predict": 100},
        },
        timeout=timeout,
    )
    response.raise_for_status()
    return json.loads(response.json()["message"]["content"])


def rewrite(question, model=MODEL):
    """Return the English question with aligned Icelandic and English keywords."""
    result = {"original": question, "ISKeywords": [], "ENKeywords": []}

    try:
        suggestions = ask_llm(question, model=model)
    except (requests.RequestException, ValueError, KeyError) as error:
        result["llm_error"] = str(error)
        return result

    icelandic_vocabulary = title_vocabulary(CATALOG_IS_PATH)
    english_vocabulary = title_vocabulary(CATALOG_EN_PATH)
    is_suggestions = suggestions.get("ISKeywords", [])
    en_suggestions = suggestions.get("ENKeywords", [])

    for is_suggestion, en_suggestion in zip(is_suggestions, en_suggestions):
        is_words = clean_keyword(is_suggestion, icelandic_vocabulary)
        en_words = clean_keyword(en_suggestion, english_vocabulary)
        if not is_words or not en_words:
            continue

        is_keyword = " ".join(is_words)
        en_keyword = " ".join(en_words)
        if (
            is_keyword in result["ISKeywords"]
            or en_keyword in result["ENKeywords"]
        ):
            continue

        result["ISKeywords"].append(is_keyword)
        result["ENKeywords"].append(en_keyword)
        if len(result["ENKeywords"]) == MAX_KEYWORDS:
            break

    return result


if __name__ == "__main__":
    questions = sys.argv[1:] or [
        "How many people are unemployed?",
        "How much does it cost to rent an apartment?",
        "How many tourists visited Iceland in 2024?",
    ]

    for question in questions:
        result = rewrite(question)
        print(question)
        print("  ISKeywords:", result["ISKeywords"], result.get("llm_error", ""))
        print("  ENKeywords:", result["ENKeywords"])
