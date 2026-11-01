"""Let an LLM rank the search candidates and pick the best tables.

The search (table_search.py) is good at getting the right table into the top N
but often ranks a near miss first. Here the LLM sees the question and the N
candidates (title, period, variable names) and answers with the numbers of
the best tables, best first (up to TOP_TABLES). The first one is used; the
others are backups, e.g. if the PX query on the first fails. If the LLM fails
or answers with something invalid, the search's own order is used.

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

from normalize import QuestionError, clean, normalize_question
from table_search import describe_candidates, latest_year, load_tables, search_tables

# Where Ollama runs. Set OLLAMA_HOST to use another machine, e.g. through an SSH tunnel.
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
if "://" not in OLLAMA_HOST:  # the ollama CLI also accepts "host:port"
    OLLAMA_HOST = "http://" + OLLAMA_HOST
OLLAMA_URL = OLLAMA_HOST + "/api/chat"
MODEL = "qwen3:30b-a3b"
CANDIDATES = 10
TOP_TABLES = 3

# The main table for common topics, keyed by word stems from the question.
# These are put first in the candidate list so the LLM always sees them.
P = "https://px.hagstofa.is/pxis/api/v1/is/"
MAIN_TABLES = [
    (("mannfj", "ibua", "ibuar", "bjugg", "bua", "byr", "folksfj", "population", "people", "live"),
     [P + "Ibuar/mannfjoldi/1_yfirlit/Yfirlit_mannfjolda/MAN00000.px",
      P + "Ibuar/mannfjoldi/1_yfirlit/Yfirlit_mannfjolda/MAN00101.px",
      P + "Ibuar/mannfjoldi/2_byggdir/sveitarfelog/MAN02005.px"]),
    (("verdbolg", "visitala neysluv", "neysluverd", "inflation", "consumer price"),
     [P + "Efnahagur/visitolur/1_vnv/1_vnv/VIS01000.px"]),
    (("flutt", "buferla", "migration", "moved", "immigra", "emigra"),
     [P + "Ibuar/buferlaflutningar/buferlaflmillilanda/MAN01400.px",
      P + "Ibuar/buferlaflutningar/buferlaflinnanlands/buferlaflinnanlands/MAN01002.px",
      P + "Ibuar/buferlaflutningar/buferlaflinnanlands/buferlaflinnanlands/MAN01201.px"]),
    (("atvinnuleys", "atvinnulaus", "unemploy"),
     [P + "Samfelag/vinnumarkadur/vinnumarkadsrannsokn/1_manadartolur/VIN00001.px",
      P + "Samfelag/vinnumarkadur/vinnumarkadsrannsokn/2_arsfjordungstolur/VIN00920.px"]),
    (("launavisit", "wage index"),
     [P + "Samfelag/launogtekjur/2_lvt/1_manadartolur/LAU04000.px"]),
    (("faedd", "faeding", "births", "born"),
     [P + "Ibuar/Faeddirdanir/Faeddir/faedingar/MAN05203.px"]),
    (("danir", "dou", "deaths", "died"),
     [P + "Ibuar/Faeddirdanir/danir/danir/MAN05210.px"]),
    (("gistin", "hotel"),
     [P + "Atvinnuvegir/ferdathjonusta/Gisting/1_hotelgistiheimili/SAM01103.px"]),
    (("ferdam", "tourist", "farthega", "keflav"),
     [P + "Atvinnuvegir/ferdathjonusta/farthegar/1_farthegar/SAM02001.px"]),
    (("landsframl", "hagvoxt", "gdp"),
     [P + "Efnahagur/thjodhagsreikningar/landsframl/1_landsframleidsla/THJ01000.px"]),
    (("ibudaverd", "fasteignaverd", "husnaedisverd", "house price", "housing price"),
     [P + "Efnahagur/visitolur/1_vnv/3_greiningarvisitolur/VIS01106.px"]),
]

SYSTEM_PROMPT = """You choose which Statistics Iceland (Hagstofa) table best answers a question.
The question is in Icelandic. You get a numbered list of tables with their title,
time period and variables. Pick the table that contains the data needed to answer
the question. Prefer tables that cover the years asked about and have the needed
breakdown (e.g. sveitarfélag, kyn, aldur). Prefer current tables over old ones.
Prefer the main, general table for a topic. Avoid methodology or revision tables
("breytt aðferðafræði", "Ofmat") and historical tables unless the question asks for them.
Answer only with JSON listing the best tables, best first: {"tables": [<number>, ...]}"""


def format_candidates(candidates):
    lines = []
    for number, candidate in enumerate(candidates, start=1):
        variables = ", ".join(v["name"] for v in candidate["variables"])
        period = f" [{candidate['period']}]" if candidate["period"] else ""
        lines.append(f"{number}. {candidate['title']}{period} | {variables}")
    return "\n".join(lines)


def candidate_tables(normalized, limit):
    """Search results, with main tables for the topic first and too-old tables removed."""
    by_url = {table["url"]: table for table in load_tables()}
    text = " ".join([clean(normalized["original"])] + normalized["keywords"]
                    + normalized["expanded"] + [clean(k) for k in normalized.get("ISKeywords", [])])
    main = [
        (by_url[url], 0.0)
        for stems, urls in MAIN_TABLES
        if any(stem in text for stem in stems)
        for url in urls if url in by_url
    ]
    results = main + search_tables(normalized, limit=limit * 3, mode="hybrid")

    # Drop tables that end before the newest year asked about (unless nothing is left).
    asked = normalized["time"]["years"]
    if asked:
        recent = [(t, s) for t, s in results if (latest_year(t) or 9999) >= max(asked)]
        results = recent or results

    seen, unique = set(), []
    for table, score in results:
        if table["url"] not in seen:
            seen.add(table["url"])
            unique.append((table, score))
    return unique[:limit]


def ask_llm(question, candidates, model=MODEL, top=TOP_TABLES, timeout=180):
    """Return the 1-based numbers the LLM picked, best first."""
    prompt = f"Question: {question}\n\nTables:\n{format_candidates(candidates)}"
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            # A JSON schema makes Ollama only produce {"tables": [<integer>, ...]}.
            "format": {
                "type": "object",
                "properties": {
                    "tables": {"type": "array", "items": {"type": "integer"}, "maxItems": top},
                },
                "required": ["tables"],
            },
            "stream": False,
            "think": False,
            "options": {"temperature": 0, "num_predict": 40, "num_ctx": 8192},
        },
        timeout=timeout,
    )
    response.raise_for_status()
    numbers = json.loads(response.json()["message"]["content"])["tables"]
    return [int(n) for n in numbers]


def choose_tables(normalized, model=MODEL, candidates=CANDIDATES, top=TOP_TABLES):
    """Return (ranked tables, all candidates, how they were chosen). Best table first."""
    options = describe_candidates(candidate_tables(normalized, candidates))
    if not options:
        return [], options, "no candidates"
    try:
        numbers = ask_llm(normalized["original"], options, model=model, top=top)
    except (requests.RequestException, ValueError, KeyError, TypeError) as error:
        return options[:top], options, f"fallback: {error}"

    picked = []
    for number in numbers:
        if 1 <= number <= len(options) and options[number - 1] not in picked:
            picked.append(options[number - 1])
    if not picked:
        return options[:top], options, f"fallback: invalid answer {numbers}"
    return picked[:top], options, "llm"


def choose_table(normalized, model=MODEL, candidates=CANDIDATES):
    """Return (best table, all candidates, how it was chosen)."""
    ranked, options, how = choose_tables(normalized, model=model, candidates=candidates)
    return (ranked[0] if ranked else None), options, how


def evaluate(model, candidates, limit=None):
    from eval_search import test_sets

    for name, questions in test_sets().items():
        items = list(questions.items())[:limit]
        search_top1 = search_top3 = llm_top1 = llm_top3 = in_candidates = fallbacks = 0
        start = time.time()

        for question, expected in items:
            try:
                normalized = normalize_question(question)
            except QuestionError:
                continue
            ranked, options, how = choose_tables(normalized, model=model, candidates=candidates)
            if not options:
                continue
            search_top1 += options[0]["id"] in expected
            search_top3 += any(o["id"] in expected for o in options[:3])
            in_candidates += any(o["id"] in expected for o in options)
            llm_top1 += ranked[0]["id"] in expected
            llm_top3 += any(o["id"] in expected for o in ranked[:3])
            fallbacks += how != "llm"

        n = len(items)
        seconds = (time.time() - start) / max(n, 1)
        print(
            f"{name:11} {model:14} N={candidates:<3} "
            f"#1 search {search_top1}/{n} vs LLM {llm_top1}/{n}   "
            f"top3 search {search_top3}/{n} vs LLM {llm_top3}/{n}   "
            f"in candidates {in_candidates}/{n}   fallbacks {fallbacks}   ({seconds:.1f} s/question)",
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
        ranked, options, how = choose_tables(normalize_question(question), args.model, args.candidates)
        print(question)
        for number, option in enumerate(options, start=1):
            rank = next((i + 1 for i, r in enumerate(ranked) if r is option), None)
            marker = f"#{rank}" if rank else "  "
            print(f" {marker} {number:2}. {option['id']:13} {option['title'][:80]}")
        print(f"   ({how})")
