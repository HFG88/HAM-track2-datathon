"""Check answerability of tables returned by table_search.search_tables().

Input:  normalize.normalize_question(...) dict and [(enriched_table, search_score), ...].
Output: ranked, explainable shortlist. No network, Ollama, or PX-API calls.

This is a *metadata suitability filter*, not a semantic guarantee. It does not
choose the codes/values for every dimension; that belongs before px_query.py.
Keep full table URLs as identifiers, since some IDs occur in multiple folders.
"""

import re
from urllib.parse import unquote

from normalize import clean, tokenize


# Each normalized breakdown maps to possible PX variable-name fragments.
# Avoid matching title text alone: the dimension must actually exist.
DIMENSION_ALIASES = {
    "kyn": ("kyn", "sex", "gender"),
    "aldur": ("aldur", "aldurs", "age"),
    "sveitarfelag": ("sveitarfel", "municipal"),
    "landsvaedi": ("landsvaed", "landshlut", "svaedi", "region"),
    "hverfi": ("hverfi", "neighborhood"),
    "rikisfang": ("rikisfang", "citizenship", "nationality"),
    "faedingarland": ("faedingarland", "country of birth"),
    "menntun": ("menntun", "education"),
    "atvinnugrein": ("atvinnugrein", "industry", "economic activity"),
}

MEASURE_ALIASES = {
    "hlutfall": ("hlutfall", "prosent", "percent", "%", "rate"),
    "kronur": ("kronur", "krona", "isk", "amount in isk"),
    "visitala": ("visitala", "visitol", "index"),
    "fjoldi": ("fjoldi", "fjolda", "number of", "count"),
}

_YEAR = re.compile(r"^(?:18|19|20)\d{2}$")
_MONTH = re.compile(r"^((?:18|19|20)\d{2})M(0?[1-9]|1[0-2])$", re.IGNORECASE)
_QUARTER = re.compile(r"^((?:18|19|20)\d{2})(?:Q|K|A|Á)([1-4])$", re.IGNORECASE)


def _time_dimension(variable):
    name = clean(str(variable.get("code", "")) + " " + str(variable.get("text", "")))
    return variable.get("time") is True or any(
        word in name for word in ("ar", "manudur", "arsfjord", "month", "quarter", "year")
        if len(word) > 2
    ) or name.strip() in {"ar", "year", "time", "timi"}


def _time_values(variables):
    """Inventory supported periods, using exact metadata codes and labels."""
    years, months, quarters = set(), set(), set()
    found_time = False
    for variable in variables:
        if not _time_dimension(variable):
            continue
        found_time = True
        for entry in (variable.get("values", []), variable.get("valueTexts", [])):
            for value in entry:
                value = str(value).strip()
                if _YEAR.fullmatch(value):
                    years.add(int(value))
                elif (match := _MONTH.fullmatch(value)):
                    months.add((int(match[1]), int(match[2])))
                elif (match := _QUARTER.fullmatch(value)):
                    quarters.add((int(match[1]), int(match[2])))
    return found_time, years, months, quarters


def _check_time(time_request, variables):
    """Return (compatible, points, reasons, warnings). Never invent aggregation."""
    if not isinstance(time_request, dict):
        time_request = {}
    requested_years = {int(y) for y in time_request.get("years", [])}
    requested_months = {int(m) for m in time_request.get("months", [])}
    requested_quarters = {int(q) for q in time_request.get("quarters", [])}
    relative = time_request.get("relative")

    requested = bool(requested_years or requested_months or requested_quarters or relative)
    has_time, years, months, quarters = _time_values(variables)
    if not requested:
        return True, 0.0, [], []
    if not has_time:
        return False, 0.0, ["No time dimension to satisfy the time request"], []

    reasons, warnings = [], []
    if relative:
        warnings.append("Relative time must be resolved to concrete periods before querying")
        return True, 0.0, reasons, warnings

    if requested_months and requested_quarters:
        return False, 0.0, ["Simultaneous month and quarter filters need explicit interpretation"], []

    if requested_months:
        candidates = months
        if not candidates:
            return False, 0.0, ["Requested monthly data unavailable"], []
        if requested_years:
            missing = {(y, m) for y in requested_years for m in requested_months} - candidates
            if missing:
                return False, 0.0, ["Missing requested month/year combinations"], []
        elif not requested_months.issubset({m for _, m in candidates}):
            return False, 0.0, ["Requested month unavailable"], []
        reasons.append("Requested monthly periods exist in metadata")
        if not requested_years:
            warnings.append("A year still needs to be chosen")
        return True, 1.5, reasons, warnings

    if requested_quarters:
        if not quarters:
            return False, 0.0, ["Requested quarterly data unavailable"], []
        if requested_years:
            missing = {(y, q) for y in requested_years for q in requested_quarters} - quarters
            if missing:
                return False, 0.0, ["Missing requested quarter/year combinations"], []
        elif not requested_quarters.issubset({q for _, q in quarters}):
            return False, 0.0, ["Requested quarter unavailable"], []
        reasons.append("Requested quarters exist in metadata")
        if not requested_years:
            warnings.append("A year still needs to be chosen")
        return True, 1.5, reasons, warnings

    if requested_years.issubset(years):
        return True, 1.5, ["Requested years available as annual periods"], []

    # A complete monthly or quarterly series could support an annual calculation,
    # but must NOT be treated as an annual value without specifying an aggregation.
    if all(all((y, m) in months for m in range(1, 13)) for y in requested_years):
        warnings.append("Only monthly data: annual result needs a justified aggregation")
        return True, -1.0, ["Full monthly coverage for requested years"], warnings
    if all(all((y, q) in quarters for q in range(1, 5)) for y in requested_years):
        warnings.append("Only quarterly data: annual result needs a justified aggregation")
        return True, -1.0, ["Full quarterly coverage for requested years"], warnings
    return False, 0.0, ["Requested years unavailable at suitable time granularity"], []


def _check_breakdowns(breakdowns, variables):
    names = [clean(str(v.get("code", "")) + " " + str(v.get("text", ""))) for v in variables]
    missing, present = [], []
    for breakdown in dict.fromkeys(breakdowns or []):
        aliases = DIMENSION_ALIASES.get(breakdown, (clean(breakdown),))
        if any(any(fragment in name for fragment in aliases) for name in names):
            present.append(breakdown)
        else:
            missing.append(breakdown)
    return present, missing


def _measure_hint(measure, table, variables):
    """Soft signal only: metadata rarely certifies the statistical measure."""
    if not measure or measure not in MEASURE_ALIASES:
        return False
    # A "how many" question isn't validated by seeing "Fjöldi" somewhere:
    # almost any unrelated statistical table may contain that label.
    if measure == "fjoldi":
        return False
    terms = MEASURE_ALIASES[measure]
    text = clean(str(table.get("title", "")))
    text += " " + " ".join(clean(str(v.get("text", ""))) for v in variables)
    # Some tables list the measure among a dimension's selectable labels.
    text += " " + " ".join(
        clean(str(label)) for variable in variables
        if not variable.get("time")
        for label in variable.get("valueTexts", [])
    )
    return any(term in text for term in terms)



# Topic signals do not certify answerability. They can, however, flag a
# suspicious table that only matches generic dimension labels (e.g. "Fjöldi").
# Keep these as a *soft* signal: semantic/vector search may find relevant tables
# with no literal title overlap.
_GENERIC_TOPIC_WORDS = {
    "fjoldi", "fjolda", "hlutfall", "prosent", "kronur", "visitala",
    "eining", "einingar", "tafla", "gogn", "tolfraedi",
}


def _topic_support(normalized, table):
    """Return (query topic terms, matching terms) from title + API folders.

    Use whole-word or conservative prefix matching, not substrings: "komu"
    shouldn't match an unrelated "afkomu" table.
    """
    terms = list(dict.fromkeys(
        token for token in (
            clean(str(t)) for t in
            (normalized.get("keywords") or []) + (normalized.get("expanded") or [])
        )
        if len(token) >= 5 and token not in _GENERIC_TOPIC_WORDS
    ))
    url = unquote(str(table.get("url", "")))
    folders = url.split("/api/v1/", 1)[-1].split("/")[1:-1] if "/api/v1/" in url else []
    words = set(tokenize(clean(str(table.get("title", "")) + " " + " ".join(folders))))
    matching = [
        term for term in terms
        if any(term == word or (len(term) >= 6 and len(word) >= 6
                                   and term[:5] == word[:5]) for word in words)
    ]
    return terms, matching


def select_tables(normalized, candidates, limit=3):
    """Filter and rerank [(enriched_table, retrieval_score), ...].

    Result: {status, selected, rejected, checked, requires_filter_resolution}.
    A `selected` entry is a candidate, *not* a verified final statistical answer.
    Every downstream PX query still needs explicit dimension-value selections.
    """
    if not isinstance(normalized, dict):
        raise TypeError("normalized must be a dict from normalize_question()")
    if not isinstance(limit, int) or limit < 1:
        raise ValueError("limit must be a positive integer")

    selected, rejected, seen_urls = [], [], set()
    for position, pair in enumerate(candidates, start=1):
        table, retrieval_score = pair
        url = table.get("url")
        if not url or url in seen_urls:
            continue
        seen_urls.add(url)
        entry = {
            "id": table.get("id"), "url": url, "title": table.get("title"),
            "retrieval_rank": position, "retrieval_score": retrieval_score,
            "reasons": [], "warnings": [],
        }
        metadata = table.get("metadata") or {}
        variables = metadata.get("variables")
        if table.get("metadata_status") not in (None, "ok") or not isinstance(variables, list) or not variables:
            entry["reasons"].append("No usable dimension metadata; cannot verify suitability")
            rejected.append(entry)
            continue

        time_ok, time_points, time_reasons, time_warnings = _check_time(
            normalized.get("time"), variables
        )
        entry["reasons"].extend(time_reasons)
        entry["warnings"].extend(time_warnings)
        matched, missing = _check_breakdowns(normalized.get("breakdowns"), variables)
        if matched:
            entry["reasons"].append("Available breakdowns: " + ", ".join(matched))
        if missing:
            entry["reasons"].append("Missing breakdowns: " + ", ".join(missing))
        if not time_ok or missing:
            rejected.append(entry)
            continue

        measure_match = _measure_hint(normalized.get("measure"), table, variables)
        if measure_match:
            entry["reasons"].append("Requested measure appears in the table metadata")
        elif normalized.get("measure"):
            entry["warnings"].append("Measure not confirmed by a simple metadata match")

        # Rank is used instead of raw retrieval scores: BM25, vector and cascade
        # have incomparable score scales. Metadata signals may adjust this order.
        score = 5.0 - (position - 1) * 0.12 + time_points
        score += 0.25 * len(matched) + (0.25 if measure_match else 0)
        topic_terms, topic_matches = _topic_support(normalized, table)
        entry["topic_matches"] = topic_matches
        if topic_terms and topic_matches:
            # A modest bonus for independent evidence in the table title/path.
            # Cap it so large questions do not overwhelm metadata checks.
            score += min(0.6, 0.3 * len(topic_matches))
        elif topic_terms:
            # Do not *reject*: vector search can retrieve true semantic matches.
            score -= 0.8
            entry["warnings"].append(
                "No clear subject-term match in table title/path; semantic review needed"
            )
        title = clean(str(table.get("title", "")))
        original = clean(str(normalized.get("original", "")))
        if re.search(r"\b\d{1,2}\s*[-–]\s*\d{1,2}\s*(?:ara|ar[a-z]*)?\b", title) and not re.search(r"\d{1,2}\s*[-–]\s*\d{1,2}", original):
            score -= 0.75
            entry["warnings"].append("Title specifies an age subgroup not requested in the question")
        entry["suitability_score"] = round(score, 3)  # heuristic, NOT probability
        selected.append(entry)

    selected.sort(key=lambda item: (-item["suitability_score"], item["retrieval_rank"]))
    return {
        "status": "shortlist" if selected else "no_verified_match",
        "selected": selected[:limit],
        "rejected": rejected,
        "checked": len(seen_urls),
        "requires_filter_resolution": True,
    }
