"""Let an LLM pick the right table from the search candidates.

The search (table_search.py) is good at getting the right table into the top N
but often ranks a near miss first. Here the LLM sees the question and the N
candidates (title, period, variable names) and answers with the number of
the best one. If the LLM fails or answers with something invalid, the
search's #1 is used.

Run from the project root:
    python table_choice.py "Hvað búa margir í Kópavogi?"
    python table_choice.py --eval --model qwen3:4b --candidates 10
"""

import argparse
import os
import json
import sys
import time

import requests

from normalize import QuestionError, normalize_question
from table_search import describe_candidates, search_tables

# Where Ollama runs. Set OLLAMA_HOST to use another machine, e.g. through an SSH tunnel.
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
if "://" not in OLLAMA_HOST:  # the ollama CLI also accepts "host:port"
    OLLAMA_HOST = "http://" + OLLAMA_HOST
OLLAMA_URL = OLLAMA_HOST + "/api/chat"
MODEL = "qwen3:4b"
CANDIDATES = 10

SYSTEM_PROMPT = """You choose which Statistics Iceland (Hagstofa) table best answers a question.
The question is in Icelandic. You get a numbered list of tables with their title,
time period and variables. Pick the table that contains the data needed to answer
the question. Prefer tables that cover the years asked about and have the needed
breakdown (e.g. sveitarfélag, kyn, aldur). Prefer current tables over old ones.
Answer only with JSON: {"table": <number>}"""


def format_candidates(candidates):
    lines = []
    for number, candidate in enumerate(candidates, start=1):
        variables = ", ".join(v["name"] for v in candidate["variables"])
        period = f" [{candidate['period']}]" if candidate["period"] else ""
        lines.append(f"{number}. {candidate['title']}{period} | {variables}")
    return "\n".join(lines)


def ask_llm(question, candidates, model=MODEL, timeout=120):
    """Return the 1-based number the LLM picked."""
    prompt = f"Question: {question}\n\nTables:\n{format_candidates(candidates)}"
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            # A JSON schema makes Ollama only produce {"table": <integer>}.
            "format": {
                "type": "object",
                "properties": {"table": {"type": "integer"}},
                "required": ["table"],
            },
            "stream": False,
            "think": False,
            "options": {"temperature": 0, "num_predict": 30, "num_ctx": 8192},
        },
        timeout=timeout,
    )
    response.raise_for_status()
    return int(json.loads(response.json()["message"]["content"])["table"])


def choose_table(normalized, model=MODEL, candidates=CANDIDATES):
    """Return (chosen candidate, all candidates, how it was chosen)."""
    options = describe_candidates(search_tables(normalized, limit=candidates, mode="hybrid"))
    if not options:
        return None, options, "no candidates"
    try:
        number = ask_llm(normalized["original"], options, model=model)
    except (requests.RequestException, ValueError, KeyError, TypeError) as error:
        return options[0], options, f"fallback: {error}"
    if not 1 <= number <= len(options):
        return options[0], options, f"fallback: invalid number {number}"
    return options[number - 1], options, "llm"


def evaluate(model, candidates, limit=None):
    from eval_search import test_sets

    for name, questions in test_sets().items():
        items = list(questions.items())[:limit]
        search_top1 = llm_top1 = in_candidates = fallbacks = 0
        start = time.time()

        for question, expected in items:
            try:
                normalized = normalize_question(question)
            except QuestionError:
                continue
            chosen, options, how = choose_table(normalized, model=model, candidates=candidates)
            if not options:
                continue
            search_top1 += options[0]["id"] in expected
            in_candidates += any(o["id"] in expected for o in options)
            llm_top1 += chosen["id"] in expected
            fallbacks += how != "llm"

        seconds = (time.time() - start) / max(len(items), 1)
        print(
            f"{name:11} {model} N={candidates:<3} "
            f"search #1: {search_top1}/{len(items)}   LLM pick: {llm_top1}/{len(items)}   "
            f"right table among candidates: {in_candidates}/{len(items)}   "
            f"fallbacks: {fallbacks}   ({seconds:.1f} s per question)",
            flush=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("questions", nargs="*")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--candidates", type=int, default=CANDIDATES)
    parser.add_argument("--eval", action="store_true", help="Measure on the test questions")
    parser.add_argument("--limit", type=int, default=None, help="Only the first N test questions")
    args = parser.parse_args()

    if args.eval:
        evaluate(args.model, args.candidates, args.limit)
        sys.exit()

    for question in args.questions or ["Hvað búa margir í Kópavogi?"]:
        chosen, options, how = choose_table(normalize_question(question), args.model, args.candidates)
        print(question)
        for number, option in enumerate(options, start=1):
            marker = "→" if option is chosen else " "
            print(f" {marker} {number:2}. {option['id']:13} {option['title'][:80]}")
        print(f"   ({how})")
