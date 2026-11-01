"""Turn a question into a structured description for table search and the PX query.

The normaliser only describes the question. It never maps anything to PX codes,
because that needs the metadata of the selected table.
"""

import json
import re
import sys
import unicodedata
from datetime import date
from functools import lru_cache
from pathlib import Path

CATALOG_PATH = Path(__file__).parent / "local" / "catalog-full.json"


class QuestionError(ValueError):
    """The question has no recognizable topic to search for."""


# Words that say nothing about which table or values are wanted.
STOPWORDS = {
    # question words
    "hver", "hvert", "hverjir", "hverjar", "hvad", "hvada", "hversu", "hvernig",
    "hvar", "hvenaer", "hve",
    # quantity words (handled by measure/intent)
    "margir", "margar", "morg", "marga", "mikill", "mikil", "mikid", "miklu",
    # verbs
    "er", "eru", "var", "voru", "verid", "vera", "hefur", "hafa", "hafi", "getur",
    # prepositions and conjunctions
    "a", "i", "ad", "af", "um", "til", "fra", "med", "vid", "fyrir", "sem", "og",
    "eda", "en", "eftir", "milli", "an", "a", "the", "of", "in",
    # time words (handled by time)
    "ar", "arid", "arinu", "arin", "ari", "arum", "sidustu", "sidasta", "sidasti",
    "nyjustu", "nyjasta", "nyjast", "fyrra", "nuna", "dag", "manud", "manudi", "manudum",
    "manudur", "manudir", "manada", "manadar",
    # rank words (handled by intent)
    "mest", "flest", "flestir", "haest", "laegst", "faest", "minnst", "staerst",
    # pronouns
    "eg", "thu", "hann", "hun", "thad", "vid", "their", "thaer", "thau", "theirra",
    # country-wide words that match too many tables
    "islandi", "island", "landinu", "tolur", "tolum",
}
ENGLISH_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "but", "by",
    "did", "do", "does", "for", "from", "had", "has", "have", "he", "her",
    "hers", "him", "his", "how", "i", "if", "in", "into", "is", "it", "its",
    "of", "on", "or", "our", "ours", "she", "that", "the", "their", "theirs",
    "them", "there", "these", "they", "this", "those", "through", "to", "us",
    "was", "we", "were", "what", "when", "where", "which", "who", "whom",
    "why", "will", "with", "you", "your", "yours", "one", "ones",
    "many", "much", "most", "more", "least", "few", "several", "some",
    "year", "years", "annual", "annually", "last", "latest", "current", "now",
    "first", "second", "third", "fourth",
}
ENGLISH_RECOGNITION_WORDS = {
    "what", "which", "who", "where", "when", "why", "how", "many", "much",
    "show", "shows", "compare", "compared", "comparison", "change", "changed",
    "increased", "decreased", "fluctuate", "fluctuated", "fluctuation",
    "population", "people", "births", "deaths", "unemployment", "inflation",
    "wages", "salary", "tourism", "tourists", "migration", "catch", "species",
    "fish", "fishing", "capelin", "industry", "age", "sex", "region",
}

# Everyday words mapped to the words used in table titles.
SYNONYMS = {
    "flutt": ["buferlaflutningar"],
    "flutti": ["buferlaflutningar"],
    "fluttu": ["buferlaflutningar"],
    "fluttust": ["buferlaflutningar"],
    "bjuggu": ["mannfjoldi"],
    "byr": ["mannfjoldi"],
    "bua": ["mannfjoldi"],
    "ibuar": ["mannfjoldi"],
    "ibua": ["mannfjoldi"],
    "ibuafjoldi": ["mannfjoldi"],
    "folksfjoldi": ["mannfjoldi"],
    "folk": ["mannfjoldi"],
    "verdbolga": ["visitala", "neysluverds"],
    "verdbolgan": ["visitala", "neysluverds"],
    "verdbolgu": ["visitala", "neysluverds"],
    "verdlag": ["visitala", "neysluverds"],
    "atvinnulausir": ["atvinnuleysi"],
    "atvinnulausra": ["atvinnuleysi"],
    "faeddust": ["faeddir"],
    "dou": ["danir"],
}

MONTHS = {
    "januar": "01", "februar": "02", "mars": "03", "april": "04",
    "mai": "05", "juni": "06", "juli": "07", "agust": "08",
    "september": "09", "oktober": "10", "november": "11", "desember": "12",
    "january": "01", "february": "02", "march": "03", "may": "05",
    "june": "06", "july": "07", "august": "08", "october": "10",
}
MONTH_ABBREVIATIONS = {
    "jan": "01", "feb": "02", "mar": "03", "apr": "04", "jun": "06", "jul": "07",
    "agu": "08", "sep": "09", "okt": "10", "nov": "11", "des": "12",
}

NUMBER_WORDS = {
    "einn": 1, "eitt": 1, "tvo": 2, "tveir": 2, "tvau": 2, "thrju": 3, "thrjú": 3,
    "thrir": 3, "fjogur": 4, "fjorir": 4, "fimm": 5, "sex": 6, "sjo": 7,
    "atta": 8, "niu": 9, "tiu": 10,
}
ENGLISH_NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}
ORDINAL_WORDS = {
    "fyrsta": 1, "fyrsti": 1, "annar": 2, "annan": 2, "odrum": 2, "annars": 2,
    "thridja": 3, "thridji": 3, "fjorda": 4, "fjordi": 4,
    "first": 1, "second": 2, "third": 3, "fourth": 4,
}

# Token prefixes that say the question wants a breakdown by that dimension.
BREAKDOWNS = {
    "kyn": ["kyn"],
    "aldur": ["aldur", "aldri", "aldurs"],
    "sveitarfelag": ["sveitarfel"],
    "landsvaedi": ["landsvaed", "landshlut"],
    "hverfi": ["hverfi", "hverfum"],
    "rikisfang": ["rikisfang"],
    "faedingarland": ["faedingarland"],
    "menntun": ["menntun"],
    "atvinnugrein": ["atvinnugrein"],
    "sex": ["sex", "gender"],
    "age": ["age"],
    "municipality": ["municipal"],
    "region": ["region", "regional"],
    "nationality": ["nationality"],
    "education": ["education"],
    "industry": ["industry"],
}

MEASURES = {
    "hlutfall": ["hlutfall", "prosent"],
    "kronur": ["kronur", "kronum", "krona", "kr", "isk"],
    "visitala": ["visitol", "visital"],
    "fjoldi": ["fjoldi", "fjolda"],
    "percentage": ["percent", "percentage", "share", "proportion"],
    "currency": ["currency", "kr", "isk"],
}

# Checked in this order: "Hvaða sveitarfélag hækkaði mest?" is a rank question.
INTENTS = [
    ("rank", ["mest", "flest", "haest", "laegst", "faest", "minnst", "staerst"]),
    ("compare", ["samanbur", "munur", "muninn", "midad"]),
    ("change", ["breyting", "breyttist", "haekk", "laekk", "aukning", "jokst",
                "minnk", "voxt", "verdbolg", "chang", "increas", "decreas",
                "fluctuat", "grew", "declin"]),
]

YEAR = r"(?:19|20)\d{2}"


def clean(text):
    """Lowercase, strip accents and map þ/ð/æ, keeping punctuation."""
    text = unicodedata.normalize("NFKD", str(text).lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    return text.replace("þ", "th").replace("ð", "d").replace("æ", "ae")


def tokenize(text):
    return re.findall(r"[a-z0-9]+", text)


def to_number(token):
    if token.isdigit():
        return int(token)
    return NUMBER_WORDS.get(token, ENGLISH_NUMBER_WORDS.get(token))


def extract_time(cleaned, tokens):
    years = []

    # Ranges: 2020-2023, frá 2020 til 2023, milli 2020 og 2023
    range_pattern = rf"({YEAR})\s*(?:-|–|—|til|og|to)\s*({YEAR})"
    for start, end in re.findall(range_pattern, cleaned):
        start, end = sorted((int(start), int(end)))
        years.extend(range(start, end + 1))
    cleaned_without_ranges = re.sub(range_pattern, " ", cleaned)
    years.extend(int(y) for y in re.findall(rf"\b{YEAR}\b", cleaned_without_ranges))

    this_year = date.today().year
    if "i fyrra" in cleaned or re.search(r"\blast year\b", cleaned):
        years.append(this_year - 1)
    if re.search(r"\b(i ar|a thessu ari|this year)\b", cleaned):
        years.append(this_year)

    months = []
    for token in tokens:
        month = next((m for name, m in MONTHS.items() if token.startswith(name)), None)
        month = month or MONTH_ABBREVIATIONS.get(token)
        if month:
            months.append(month)

    quarters = [int(q) for q in re.findall(r"\b([1-4])\.?\s*arsfjord", cleaned)]
    quarters += [int(q) for q in re.findall(r"\bq([1-4])\b", cleaned)]
    for i, token in enumerate(tokens[:-1]):
        if token in ORDINAL_WORDS and tokens[i + 1].startswith("arsfjord"):
            quarters.append(ORDINAL_WORDS[token])

    relative = None
    for i, token in enumerate(tokens):
        if token not in ("sidustu", "sidasta", "sidasti", "last"):
            continue
        rest = tokens[i + 1:i + 3]
        count = to_number(rest[0]) if rest else None
        unit_word = rest[1] if count is not None and len(rest) > 1 else (rest[0] if rest else "")
        if token != "sidustu" and count is None:
            count = 1
        if unit_word.startswith(("arsfjord", "quarter")):
            relative = {"last": count, "unit": "quarter"}
        elif unit_word.startswith(("manud", "manad", "month")):
            relative = {"last": count, "unit": "month"}
        elif unit_word.startswith(("ar", "year")):
            relative = {"last": count, "unit": "year"}

    if relative is None and not years and re.search(
        r"\b(nyjust|nyjast|nuverandi|nuna|latest|current|now)\b", cleaned
    ):
        relative = "latest"

    return {
        "years": sorted(set(years)),
        "months": sorted(set(months)),
        "quarters": sorted(set(quarters)),
        "relative": relative,
    }


def is_time_token(token):
    return (
        re.fullmatch(YEAR, token) is not None
        or token in MONTH_ABBREVIATIONS
        or any(token.startswith(name) for name in MONTHS)
        or token.startswith("arsfjord")
        or token in ORDINAL_WORDS
        or token in {"first", "second", "third", "fourth"}
    )


def find_prefixed(tokens, table):
    found = []
    for name, prefixes in table.items():
        if any(t.startswith(p) for t in tokens for p in prefixes):
            found.append(name)
    return found


def extract_measure(cleaned, tokens):
    if "%" in cleaned or any(t in {"percent", "percentage", "share", "proportion"}
                             for t in tokens):
        return "hlutfall"
    for name, prefixes in MEASURES.items():
        if any(t == p or (len(p) > 3 and t.startswith(p)) for t in tokens for p in prefixes):
            return name
    if re.search(r"\b(hversu|hve) (margir|margar|morg)\b", cleaned):
        return "fjoldi"
    if "number" in tokens or "count" in tokens:
        return "fjoldi"
    return None


def extract_intent(tokens):
    for intent, prefixes in INTENTS:
        if any(t.startswith(p) for t in tokens for p in prefixes):
            return intent
    return "value"


# Full word forms for the prefix lists above, so they count as recognized terms.
EXTRA_WORDS = {
    "kyni", "kynjum", "aldri", "aldurs", "aldursflokkum", "sveitarfelogum",
    "sveitarfelagi", "landsvaedum", "landshlutum", "hverfum", "rikisfangi",
    "menntun", "atvinnugreinum", "hlutfall", "prosent", "kronur", "visitala",
    "fjoldi", "fjolda", "breyting", "breyttist", "haekkun", "haekkadi", "laekkun",
    "laekkadi", "aukning", "jokst", "minnkadi", "minnkun", "voxtur", "samanburdur",
    "munur", "midad", "arsfjordungi", "arsfjordungur", "staekkadi",
}


@lru_cache(maxsize=1)
def vocabulary():
    """Words used to decide whether a question contains a recognizable topic."""
    words = set(STOPWORDS) | set(SYNONYMS) | set(MONTHS) | set(MONTH_ABBREVIATIONS)
    words |= set(NUMBER_WORDS) | set(ORDINAL_WORDS) | EXTRA_WORDS
    words |= {w for synonyms in SYNONYMS.values() for w in synonyms}
    if CATALOG_PATH.exists():
        for table in json.loads(CATALOG_PATH.read_text(encoding="utf-8")):
            words.update(tokenize(clean(table["title"])))
    return frozenset(w for w in words if not w.isdigit())


def normalize_question(question):
    cleaned = clean(question)
    tokens = tokenize(cleaned)

    time = extract_time(cleaned, tokens)

    keywords = []
    for token in tokens:
        if (
            token in STOPWORDS
            or token in ENGLISH_STOPWORDS
            or token in NUMBER_WORDS
            or is_time_token(token)
        ):
            continue
        if len(token) < 2:
            continue
        if token not in keywords:
            keywords.append(token)

    phrases = [f"{a} {b}" for a, b in zip(keywords, keywords[1:])]

    expanded = []
    for word in keywords:
        for synonym in SYNONYMS.get(word, []):
            if synonym not in keywords and synonym not in expanded:
                expanded.append(synonym)

    # Unknown words such as place names are allowed, but at least one keyword
    # must be recognised, otherwise there is nothing to search for.
    if (
        not any(k in vocabulary() for k in keywords + expanded)
        and not any(token in ENGLISH_RECOGNITION_WORDS for token in tokens)
    ):
        raise QuestionError(
            "Ég skildi ekki spurninguna. Athugaðu stafsetninguna og reyndu aftur."
        )

    return {
        "original": question,
        "text": " ".join(re.sub(r"[^a-z0-9%\-\s]", " ", cleaned).split()),
        "keywords": keywords,
        "phrases": phrases,
        "expanded": expanded,
        "time": time,
        "breakdowns": find_prefixed(tokens, BREAKDOWNS),
        "measure": extract_measure(cleaned, tokens),
        "intent": extract_intent(tokens),
        "corrections": {},
    }


if __name__ == "__main__":
    questions = sys.argv[1:] or [
        "Hversu margir fluttu frá Austurlandi eftir kyni 2020-2023?",
        "Hver var verðbólgan í mars 2024?",
        "Mannfjöldi á Norðurlandi eystra síðustu 5 ár",
        "Hvaða sveitarfélag stækkaði mest í fyrra?",
        "Hlutfall atvinnulausra á 2. ársfjórðungi 2025",
        "Nýjustu tölur um íbúa Reykjavíkur eftir aldri",
    ]
    for question in questions:
        try:
            print(json.dumps(normalize_question(question), ensure_ascii=False, indent=2))
        except QuestionError as error:
            print(f"{question!r}: {error}")
