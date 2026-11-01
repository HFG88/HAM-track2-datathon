"""Answer a question end to end.

question -> normalize + LLM keywords (question_router) -> search (top 10)
-> LLM ranks tables -> selections -> px_query (PX-API data)
-> LLM writes the answer in the question's language (Icelandic or English).

build_selections() turns the normalised question into a value for every
dimension of the chosen table, which px_query.query_table() requires:
- time: the years/months/quarters asked about, or the newest period
- dimensions the question breaks down by ("eftir kyni"): all values
- dimensions where a question word matches a value ("Kópavogi" ~ "Kópavogsbær"): those values
- everything else: the total ("Alls"), or the first value

Run from the project root (OLLAMA_HOST can point at another machine):
    python answer.py "Hvað búa margir í Kópavogi?"

Models can be set with environment variables:
    REWRITE_MODEL (keywords), TABLE_MODEL (table choice), ANSWER_MODEL (answer)
"""

import argparse
import json
import os
import re
import time

import requests

from normalize import BREAKDOWNS, QuestionError, clean, normalize_question, tokenize
from px_query import PXQueryError, find_table, query_table
from question_router import detect_language, rewrite_question
from table_choice import choose_tables
from table_search import is_time_variable

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
if "://" not in OLLAMA_HOST:  # the ollama CLI also accepts "host:port"
    OLLAMA_HOST = "http://" + OLLAMA_HOST
OLLAMA_URL = OLLAMA_HOST + "/api/chat"
# One model for every step, so the GPU never has to swap models between steps.
# qwen3:32b wrote short, correct answers; qwen3:30b-a3b often rambled.
MODEL = os.environ.get("MODEL", "qwen3:32b")
REWRITE_MODEL = os.environ.get("REWRITE_MODEL", MODEL)
TABLE_MODEL = os.environ.get("TABLE_MODEL", MODEL)
ANSWER_MODEL = os.environ.get("ANSWER_MODEL", MODEL)

# English breakdown names from normalize.py -> the Icelandic ones used in table variables.
ENGLISH_BREAKDOWNS = {
    "sex": "kyn", "age": "aldur", "municipality": "sveitarfelag", "region": "landsvaedi",
    "nationality": "rikisfang", "education": "menntun", "industry": "atvinnugrein",
}

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

ENGLISH_INSTRUCTION = (
    "The question is in English: write the answer in English, and start the "
    'last line with "Source:" instead of "Heimild:".'
)


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
    labels = variable.get("valueTexts") or codes
    # Years are in the codes (2024M03) or, in many tables, only in the labels (code "13" = "2017").
    periods = [(code, parse_period(code) or parse_period(label.strip())) for code, label in zip(codes, labels)]
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
        name = dict(zip(codes, labels))
        available = f"{name[periods[0][0]]}–{name[newest]}"
        return [newest], f"Umbeðið tímabil er ekki í töflunni (hún nær yfir {available}); notað er nýjasta tímabilið, {name[newest]}."
    if not months and not quarters:
        # Only years asked about in a monthly/quarterly table: take each year's last period.
        parsed = dict(periods)
        chosen = last_period_per_year([(c, parsed[c]) for c in chosen], years)
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

    # Whole-word matches first: "Flateyri" should not also pick "Flatey á Breiðafirði".
    exact = [
        code for code, label in zip(variable["values"], variable["valueTexts"])
        if set(words) & set(tokenize(clean(label)))
    ]
    if exact:
        return exact

    matches = []
    for code, label in zip(variable["values"], variable["valueTexts"]):
        tokens = tokenize(clean(label))
        cleaned = " ".join(tokens)
        if (
            any(p in cleaned for p in phrases)
            or any(t[:5] in stems for t in tokens if len(t) >= 4)
            # Compounds: "fluttu" ~ "Brottfluttir", "atvinnulausir" ~ "Atvinnulausir alls"
            or any(stem in t for stem in stems for t in tokens if len(t) > len(stem) + 2)
        ):
            matches.append(code)
    return matches


def is_breakdown(variable, normalized):
    name = clean(variable.get("text", ""))
    for breakdown in normalized["breakdowns"]:
        breakdown = ENGLISH_BREAKDOWNS.get(breakdown, breakdown)
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


def place_names(normalized):
    """Capitalised words in the question (not the first word), e.g. Austurlandi, Kópavogi."""
    country = {"island", "islandi", "islands", "iceland", "icelandic"}
    words = re.findall(r"\w+", normalized["original"])[1:]
    return [clean(w) for w in words if w[:1].isupper() and not w.isupper() and clean(w) not in country]


def unmatched_place_notes(normalized, table, selections):
    """Notes for place names that no selected value matches, so the answer can't claim them."""
    selected_labels = " ".join(
        clean(label)
        for variable in table["metadata"]["variables"]
        for code, label in zip(variable["values"], variable["valueTexts"])
        if code in selections.get(variable["code"], [])
    )
    title = clean(table["title"])
    notes = []
    for place in place_names(normalized):
        stem = place[:5]
        if stem not in selected_labels and stem not in title:
            notes.append(
                f"'{place}' fannst ekki í töflunni; tölurnar eiga EKKI sérstaklega við það, "
                "heldur við heildina sem valin er. Segðu það skýrt í svarinu."
            )
    return notes


def build_selections(normalized, table):
    """Return ({dimension code: [value codes]}, [notes in Icelandic])."""
    selections, notes = {}, []
    for variable in table["metadata"]["variables"]:
        code = variable["code"]
        if is_time_variable(variable):
            values, note = select_time(variable, normalized["time"], normalized["intent"])
            if note:
                notes.append(note)
        elif is_breakdown(variable, normalized) and len(variable["values"]) <= MAX_BREAKDOWN_VALUES:
            values = list(variable["values"])
        else:
            values = matching_values(variable, normalized)[:MAX_MATCHED_VALUES]
            others = [c for c, l in zip(variable["values"], variable["valueTexts"]) if not is_total(l)]
            if len(others) < len(variable["values"]) and set(others) <= set(values):
                # The word matches every part ("börn" ~ "Börn giftra/ógiftra mæðra"),
                # so it doesn't narrow anything down: use the total.
                values = []
            if not values:
                values = [default_value(variable, normalized["intent"])]
        selections[code] = values
    notes += unmatched_place_notes(normalized, table, selections)
    return selections, notes


def shrink(selections, table):
    """If the query is too big, keep only the newest time period."""
    for variable in table["metadata"]["variables"]:
        if is_time_variable(variable) and len(selections[variable["code"]]) > 1:
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

def ask_for_answer(context, system, model, timeout):
    """One LLM call; returns the answer text (or the raw output if it isn't valid JSON)."""
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": system},
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
            "options": {"temperature": 0.2, "num_predict": 350, "repeat_penalty": 1.15,
                        "num_ctx": 8192},
        },
        timeout=timeout,
    )
    response.raise_for_status()
    text = response.json()["message"]["content"]
    try:
        return json.loads(text)["svar"].strip()
    except (ValueError, KeyError):
        if text.lstrip().startswith('{"svar"'):
            # Cut off mid-answer: keep the complete sentences.
            body = text.split(":", 1)[1].strip().lstrip('"')
            return body[:body.rfind(".") + 1] or body
        # Some Qwen builds still write their reasoning; keep only what follows it.
        return text.split("</think>")[-1].strip()


def echoes_rows(answer):
    """True if the model copied the data lines ("Ár: 2023 | Kyn: … → … = 1311") instead of answering."""
    return "→" in answer and answer.count("|") >= 2


def write_answer(question, table, selections_text, notes, rows, model=ANSWER_MODEL,
                 language="is", timeout=180):
    context = "\n".join([
        f"Spurning: {question}",
        f"Tafla: {table['title']}",
        f"Slóð: {table['url']}",
        f"Valdar víddir: {selections_text}",
        *(f"Athugasemd: {note}" for note in notes),
        "Gögn:",
        *rows,
    ])
    system = ANSWER_PROMPT + ("\n" + ENGLISH_INSTRUCTION if language == "en" else "")
    answer = ask_for_answer(context, system, model, timeout)
    if echoes_rows(answer):
        retry = ("\nEkki afrita gagnalínurnar. Skrifaðu eina eða tvær heilar setningar sem "
                 "svara spurningunni með tölunum, og leggðu saman ef spurt er um heildina.")
        answer = ask_for_answer(context, system + retry, model, timeout)
    if echoes_rows(answer):
        raise ValueError("the model repeated the data rows instead of answering")
    return answer


def answer_question(question, rewrite_model=REWRITE_MODEL, table_model=TABLE_MODEL,
                    answer_model=ANSWER_MODEL):
    """Run the whole pipeline. Returns a dict with every step, for debugging and the frontend."""
    timings = {}
    start = time.time()
    try:
        if detect_language(question) == "en":
            # English needs the LLM to translate keywords into Icelandic (ISKeywords).
            normalized = rewrite_question(question, model=rewrite_model)
        else:
            # Icelandic: the rule-based normaliser is enough and saves an LLM call.
            normalized = normalize_question(question)
            normalized["language"] = "is"
    except QuestionError as error:
        return {"question": question, "answer": str(error), "error": "question"}
    except (requests.RequestException, ValueError, KeyError):
        # The keyword LLM failed: carry on with the rule-based normaliser alone.
        try:
            normalized = normalize_question(question)
        except QuestionError as error:
            return {"question": question, "answer": str(error), "error": "question"}
    language = normalized.get("language", "is")
    timings["rewrite"] = round(time.time() - start, 2)

    ranked, candidates, how = choose_tables(normalized, model=table_model)
    timings["table_choice"] = round(time.time() - start - timings["rewrite"], 2)
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
    timings["px_query"] = round(time.time() - start - sum(timings.values()), 2)

    rows = data_rows(result, table)
    chosen_selections = selection_text(selections, table)
    try:
        answer = write_answer(question, table, chosen_selections, notes, rows,
                              model=answer_model, language=language)
        answer_by = "llm"
        if "Heimild" not in answer and "Source" not in answer:
            label = "Source" if language == "en" else "Heimild"
            answer += f"\n\n{label}: {table['title']}, {table['url']}"
    except (requests.RequestException, ValueError) as error:
        # The LLM is down or failed: show the numbers without prose rather than nothing.
        answer = "\n".join(notes + rows[:10] + [f"Heimild: {table['title']}, {table['url']}"])
        answer_by = f"template ({error})"
    timings["answer"] = round(time.time() - start - sum(timings.values()), 2)
    timings["total"] = round(time.time() - start, 2)

    return {
        "question": question,
        "language": language,
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


MAX_QUESTIONS = 5


def split_questions(text):
    """Split a prompt into separate questions: after each "?" and on new lines."""
    parts = re.split(r"(?<=\?)\s+|\n+", text.strip())
    return [p.strip() for p in parts if len(p.strip()) > 3][:MAX_QUESTIONS] or [text.strip()]


def answer_questions(text, **models):
    """Answer every question in the prompt; returns one result per question."""
    return [answer_question(question, **models) for question in split_questions(text)]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("questions", nargs="*")
    parser.add_argument("--json", action="store_true", help="Print the full result as JSON")
    args = parser.parse_args()

    for question in args.questions or ["Hvað búa margir í Kópavogi?"]:
        result = answer_question(question)
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
