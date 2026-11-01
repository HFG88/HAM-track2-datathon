"""Answer a question end to end.

question -> normalize -> search (top 10) -> LLM ranks tables -> selections
-> px_query (PX-API data) -> LLM writes the answer in Icelandic.

build_selections() turns the normalised question into a value for every
dimension of the chosen table, which px_query.query_table() requires:
- time: the years/months/quarters asked about, or the newest period
- dimensions the question breaks down by ("eftir kyni"): all values
- dimensions where a question word matches a value ("Kópavogi" ~ "Kópavogsbær"): those values
- everything else: the total ("Alls"), or the first value

Run from the project root (OLLAMA_HOST can point at another machine):
    python answer.py "Hvað búa margir í Kópavogi?"
"""

import argparse
import json
import os
import re
import time

import requests

from normalize import BREAKDOWNS, QuestionError, clean, normalize_question, tokenize
from px_query import PXQueryError, find_table, query_table
from table_choice import choose_tables

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
if "://" not in OLLAMA_HOST:  # the ollama CLI also accepts "host:port"
    OLLAMA_HOST = "http://" + OLLAMA_HOST
OLLAMA_URL = OLLAMA_HOST + "/api/chat"
MODEL = "qwen3:30b-a3b"

MAX_CELLS = 500
MAX_MATCHED_VALUES = 10
MAX_BREAKDOWN_VALUES = 30
MAX_DATA_ROWS = 80

TOTAL_LABELS = ("alls", "samtals", "allir", "allt landid", "landid allt", "heild")

ANSWER_PROMPT = """Þú ert Hagur, aðstoðarmaður Hagstofu Íslands.
Svaraðu spurningu notandans á íslensku, stutt og skýrt, EINGÖNGU út frá gögnunum sem fylgja.
- Nefndu tölurnar með einingu (t.d. fjöldi, %, krónur, vísitölustig) og tímabilinu sem þær eiga við.
- Ef gögnin svara spurningunni ekki að fullu, segðu það hreint út og hvað vantar. Ekki giska.
- Ef athugasemdir fylgja (t.d. að umbeðið ár sé ekki til), taktu það fram.
- Endaðu á línu sem byrjar á "Heimild:" með heiti töflunnar og slóðinni."""


# ---------------------------------------------------------------- selections

def parse_period(code):
    """'2024' -> (2024, None, None), '2024M03' -> (2024, 3, None), '2010Á4' -> (2010, None, 4)."""
    match = re.match(r"^(\d{4})(?:M(\d{2})|\D?(\d))?$", code)
    if not match:
        return None
    year, month, quarter = match.groups()
    return int(year), int(month) if month else None, int(quarter) if quarter else None


def select_time(variable, time_info, intent):
    """Return (value codes, note or None) for the time dimension."""
    codes = variable["values"]
    periods = [(code, parse_period(code)) for code in codes]
    periods = [(code, p) for code, p in periods if p]
    if not periods:
        return [codes[-1]], None
    newest = periods[-1][0]

    relative = time_info["relative"]
    if isinstance(relative, dict):
        count = relative.get("last") or 5
        if relative["unit"] == "year":
            years = sorted({p[0] for _, p in periods})[-count:]
            return last_period_per_year(periods, years), None
        return [code for code, _ in periods[-count:]], None
    if relative == "latest" or not time_info["years"]:
        return [newest], None

    years = set(time_info["years"])
    if intent == "change" and len(years) == 1:
        years.add(min(years) - 1)  # a change needs something to compare with
    months, quarters = set(map(int, time_info["months"])), set(time_info["quarters"])

    chosen = [
        code for code, (year, month, quarter) in periods
        if year in years
        and (not months or month is None or month in months)
        and (not quarters or quarter is None or quarter in quarters)
    ]
    if not chosen:
        available = f"{periods[0][0]}–{newest}"
        return [newest], f"Umbeðið tímabil er ekki í töflunni (hún nær yfir {available}); notað er nýjasta tímabilið, {newest}."
    if not months and not quarters:
        # Only years asked about in a monthly/quarterly table: take each year's last period.
        chosen = last_period_per_year([(c, parse_period(c)) for c in chosen], years)
    return chosen, None


def last_period_per_year(periods, years):
    last = {}
    for code, (year, _, _) in periods:
        if year in years:
            last[year] = code
    return [last[year] for year in sorted(last)]


def is_total(label):
    text = " ".join(tokenize(clean(label)))
    return any(text == t or text.startswith(t + " ") for t in TOTAL_LABELS)


def matching_values(variable, normalized):
    """Values whose label contains a word from the question (by 5-letter stem)."""
    words = [w for w in normalized["keywords"] if len(w) >= 4 and not w.isdigit()]
    stems = {w[:5] for w in words}
    phrases = normalized["phrases"]
    matches = []
    for code, label in zip(variable["values"], variable["valueTexts"]):
        tokens = tokenize(clean(label))
        cleaned = " ".join(tokens)
        if any(p in cleaned for p in phrases) or any(t[:5] in stems for t in tokens if len(t) >= 4):
            matches.append(code)
    return matches


def is_breakdown(variable, normalized):
    name = clean(variable.get("text", ""))
    for breakdown in normalized["breakdowns"]:
        if any(prefix in name for prefix in BREAKDOWNS.get(breakdown, [breakdown])):
            return True
    return False


def default_value(variable, intent):
    labels = variable["valueTexts"]
    if intent == "change":
        for code, label in zip(variable["values"], labels):
            if "arsbreyting" in clean(label).replace(" ", ""):
                return code
    for code, label in zip(variable["values"], labels):
        if is_total(label):
            return code
    return variable["values"][0]


def build_selections(normalized, table):
    """Return ({dimension code: [value codes]}, [notes in Icelandic])."""
    selections, notes = {}, []
    for variable in table["metadata"]["variables"]:
        code = variable["code"]
        if variable.get("time") or code in ("Ár", "Mánuður", "Ársfjórðungur"):
            values, note = select_time(variable, normalized["time"], normalized["intent"])
            if note:
                notes.append(note)
        elif is_breakdown(variable, normalized) and len(variable["values"]) <= MAX_BREAKDOWN_VALUES:
            values = list(variable["values"])
        else:
            values = matching_values(variable, normalized)[:MAX_MATCHED_VALUES]
            if not values:
                values = [default_value(variable, normalized["intent"])]
        selections[code] = values
    return selections, notes


def shrink(selections, table):
    """If the query is too big, keep only the newest time period."""
    for variable in table["metadata"]["variables"]:
        if variable.get("time") and len(selections[variable["code"]]) > 1:
            selections[variable["code"]] = selections[variable["code"]][-1:]
    return selections


# ---------------------------------------------------------------- data -> text

def data_rows(result, table):
    """Turn the PX response into readable lines: 'Mánuður: 2024M03 | Liður: Ársbreyting, % = 6.8'."""
    labels = {
        v["code"]: dict(zip(v["values"], v["valueTexts"]))
        for v in table["metadata"]["variables"]
    }
    key_columns = [c for c in result["columns"] if c["type"] in ("d", "t")]
    value_columns = [c for c in result["columns"] if c["type"] == "c"]
    rows = []
    for item in result["data"][:MAX_DATA_ROWS]:
        keys = [
            f"{column['text']}: {labels.get(column['code'], {}).get(key, key).strip()}"
            for column, key in zip(key_columns, item["key"])
        ]
        values = [f"{column['text']} = {value}" for column, value in zip(value_columns, item["values"])]
        rows.append(" | ".join(keys) + " → " + "; ".join(values))
    return rows


def selection_text(selections, table):
    labels = {
        v["code"]: (v["text"], dict(zip(v["values"], v["valueTexts"])))
        for v in table["metadata"]["variables"]
    }
    parts = []
    for code, values in selections.items():
        name, names = labels[code]
        shown = [names.get(v, v).strip() for v in values]
        parts.append(f"{name}: {', '.join(shown[:6])}{' …' if len(shown) > 6 else ''}")
    return "; ".join(parts)


# ---------------------------------------------------------------- answer

def write_answer(question, table, selections_text, notes, rows, model=MODEL, timeout=180):
    context = "\n".join([
        f"Spurning: {question}",
        f"Tafla: {table['title']}",
        f"Slóð: {table['url']}",
        f"Valdar víddir: {selections_text}",
        *(f"Athugasemd: {note}" for note in notes),
        "Gögn:",
        *rows,
    ])
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": ANSWER_PROMPT},
                # "/no_think" is Qwen 3's switch for answering without reasoning first.
                {"role": "user", "content": context + "\n/no_think"},
            ],
            # A JSON schema stops "thinking" models from writing out their reasoning.
            "format": {
                "type": "object",
                "properties": {"svar": {"type": "string"}},
                "required": ["svar"],
            },
            "stream": False,
            "think": False,
            "options": {"temperature": 0.2, "num_predict": 600, "num_ctx": 8192},
        },
        timeout=timeout,
    )
    response.raise_for_status()
    text = response.json()["message"]["content"]
    try:
        return json.loads(text)["svar"].strip()
    except (ValueError, KeyError):
        # Some Qwen builds still write their reasoning; keep only what follows it.
        return text.split("</think>")[-1].strip()


def answer_question(question, model=MODEL):
    """Run the whole pipeline. Returns a dict with every step, for debugging and the frontend."""
    timings = {}
    start = time.time()
    try:
        normalized = normalize_question(question)
    except QuestionError as error:
        return {"question": question, "answer": str(error), "error": "question"}

    ranked, candidates, how = choose_tables(normalized, model=model)
    timings["table_choice"] = round(time.time() - start, 2)
    if not ranked:
        return {"question": question, "answer": "Ég fann enga töflu sem passar við spurninguna.",
                "error": "no_table"}

    # Try the LLM's tables in order until one gives data.
    errors = []
    for choice in ranked:
        table = find_table(choice["url"])
        selections, notes = build_selections(normalized, table)
        try:
            try:
                result = query_table(table["url"], selections, max_cells=MAX_CELLS)
            except PXQueryError as error:
                if "cells" not in str(error):
                    raise
                result = query_table(table["url"], shrink(selections, table), max_cells=MAX_CELLS)
            break
        except PXQueryError as error:
            errors.append(f"{choice['id']}: {error}")
    else:
        return {"question": question, "answer": "Ekki tókst að sækja gögn frá Hagstofunni.",
                "error": "px", "details": errors}
    timings["px_query"] = round(time.time() - start - timings["table_choice"], 2)

    rows = data_rows(result, table)
    chosen_selections = selection_text(selections, table)
    try:
        answer = write_answer(question, table, chosen_selections, notes, rows, model=model)
        answer_by = "llm"
    except (requests.RequestException, ValueError) as error:
        # The LLM is down or failed: show the numbers without prose rather than nothing.
        answer = "\n".join(notes + rows[:10] + [f"Heimild: {table['title']}, {table['url']}"])
        answer_by = f"template ({error})"
    timings["answer"] = round(time.time() - start - sum(timings.values()), 2)

    return {
        "question": question,
        "answer": answer,
        "table": {"id": table["id"], "title": table["title"], "url": table["url"]},
        "table_choice": how,
        "answer_by": answer_by,
        "candidates": [{"id": c["id"], "title": c["title"]} for c in candidates],
        "selections": chosen_selections,
        "notes": notes,
        "data": rows,
        "normalized": normalized,
        "timings": timings,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("questions", nargs="*")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--json", action="store_true", help="Print the full result as JSON")
    args = parser.parse_args()

    for question in args.questions or ["Hvað búa margir í Kópavogi?"]:
        result = answer_question(question, model=args.model)
        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
            continue
        print(f"\n### {question}")
        if "table" in result:
            print(f"Tafla:   {result['table']['id']} {result['table']['title']}")
            print(f"Víddir:  {result['selections']}")
            for row in result["data"][:5]:
                print(f"Gögn:    {row}")
            print(f"Tími:    {result['timings']}")
        print(f"\n{result['answer']}")
