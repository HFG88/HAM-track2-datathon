"""Propose PX dimension filters with Ollama; validate against cached metadata.

The model cannot execute queries. The caller may call px_query.query_table ONLY
when the returned status is 'ready'. Never invent a dimension default or run a
partial/oversized query. Table identity is the exact catalogue URL.

This is a deliberately small hackathon prototype, not a general statistical
calculation planner. Multiple tables and derived measures require other steps.
"""

import json
import re
import unicodedata

import requests

from px_query import (
    DEFAULT_CATALOG, DEFAULT_MAX_CELLS, PXQueryError,
    find_table, prepare_query,
)

DEFAULT_MODEL = "qwen3:30b-instruct"
DEFAULT_ENDPOINT = "http://127.0.0.1:11435/api/chat"
MAX_OPTIONS_PER_DIMENSION = 80


class FilterResolutionError(RuntimeError):
    """The LLM could not supply a valid structured filter proposal."""


def _fold(value):
    text = unicodedata.normalize("NFKD", str(value).casefold())
    return "".join(ch for ch in text if not unicodedata.combining(ch))


def _choice_summary(variable, question, requested_years, *, limit=MAX_OPTIONS_PER_DIMENSION):
    """Present genuine code/label pairs; shortlist large dimensions conservatively."""
    codes = variable.get("values", [])
    labels = variable.get("valueTexts", [])
    if (not isinstance(codes, list) or not isinstance(labels, list)
            or len(codes) != len(labels)):
        raise FilterResolutionError(f"Malformed metadata for {variable.get('code')!r}")

    pairs = [{"code": str(code), "label": str(label)}
             for code, label in zip(codes, labels)]
    if len(pairs) <= limit:
        return pairs, False

    q = _fold(question)
    selected = []
    for pair in pairs:
        code, label = pair["code"], pair["label"]
        folded = _fold(label)
        is_total = folded in {"alls", "samtals", "allir", "allar tegundir", "heildarafli", "farþegar alls"}
        is_explicit = len(folded) >= 3 and folded in q
        is_requested_year = any(re.match(rf"^{re.escape(y)}(?:$|\D)", code)
                                or re.match(rf"^{re.escape(y)}(?:$|\D)", label)
                                for y in requested_years)
        is_latest = ("nyjust" in q or "latest" in q) and pair is pairs[-1]
        if is_total or is_explicit or is_requested_year or is_latest:
            selected.append(pair)
    return selected[:limit], True


def _schema():
    return {
        "type": "object",
        "properties": {
            "selections": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "dimension": {"type": "string"},
                        "values": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["dimension", "values"],
                    "additionalProperties": False,
                },
            },
            "unresolved": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "dimension": {"type": "string"},
                        "reason": {"type": "string"},
                    },
                    "required": ["dimension", "reason"],
                    "additionalProperties": False,
                },
            },
            "reason": {"type": "string"},
        },
        "required": ["selections", "unresolved", "reason"],
        "additionalProperties": False,
    }


def resolve_filters(
    question, table_url, *, normalized=None, catalog_path=DEFAULT_CATALOG,
    model=DEFAULT_MODEL, endpoint=DEFAULT_ENDPOINT, max_cells=DEFAULT_MAX_CELLS,
    timeout=120, session=None,
):
    """Return validated filters or a clarification; never call Statistics Iceland.

    Args:
      question: Original natural-language question.
      table_url: Exact selected catalogue URL (not ambiguous table ID).
      normalized: Optional output of normalize_question; used for requested years.
      catalog_path: Enriched catalogue snapshot path.
      session: Optional requests-compatible HTTP client (test injection).

    A 'ready' result's `selections` maps dimension code to exact API code(s),
    and is safe to pass to px_query.query_table. 'needs_clarification' is not.
    """
    table = find_table(table_url, catalog_path=catalog_path)
    variables = table["metadata"]["variables"]
    if not variables:
        raise FilterResolutionError("Table has no dimensions")
    requested_years = [str(y) for y in ((normalized or {}).get("time") or {}).get("years", [])]
    prompts = []
    shown = {}
    for var in variables:
        code = var["code"]
        choices, truncated = _choice_summary(var, question, requested_years)
        shown[code] = {choice["code"] for choice in choices}
        prompts.append({
            "dimension": code,
            "label": var.get("text", code),
            "is_time": bool(var.get("time")),
            "options": choices,
            "options_truncated": truncated,
            "total_option_count": len(var.get("values", [])),
        })

    system_prompt = (
        "You are selecting exact Statistics Iceland PX-API dimension values. "
        "'Fiskeldi' means aquaculture, NOT wild-caught fish. "
        "For EACH dimension in the provided table, select the value code(s) "
        "that the question clearly requires, using ONLY codes shown in options. "
        "If a dimension is ambiguous, has no matching options, or requires an "
        "unspecified time period, put it in unresolved instead of guessing. "
        "Prefer an explicit aggregate such as 'Samtals'/'Farþegar alls' ONLY "
        "when the question requests the total rather than a subgroup. "
        "Never automatically choose the first option. "
        "Do not confuse distinct measures, units, populations, years, or geography. "
        "For comparisons or growth calculations, return all explicitly needed "
        "periods; do not calculate a result. "
        "A bare year does not justify choosing one arbitrary month. "
        "If the table has no appropriate option, mark the dimension unresolved. "
        "Write a short reason and valid JSON fitting the schema."
    )
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps({
                "question": question,
                "requested_years": requested_years,
                "table_title": table.get("title"),
                "table_url": table["url"],
                "dimensions": prompts,
            }, ensure_ascii=False)},
        ],
        "stream": False,
        "format": _schema(),
        "think": False,
        "options": {"temperature": 0, "num_ctx": 8192, "num_predict": 512},
    }

    http = session if session is not None else requests
    try:
        response = http.post(endpoint, json=payload, timeout=timeout)
        response.raise_for_status()
        raw = response.json()
        if raw.get("done_reason") == "length":
            raise FilterResolutionError("Ollama stopped at its generation limit; no filters accepted")
        proposal = json.loads(raw["message"]["content"])
    except (requests.RequestException, KeyError, TypeError, ValueError) as exc:
        raise FilterResolutionError(f"Ollama filter request/response failed: {exc}") from exc

    if not isinstance(proposal, dict):
        raise FilterResolutionError("Ollama returned a non-object proposal")
    if (not isinstance(proposal.get("selections"), list)
            or not isinstance(proposal.get("unresolved"), list)
            or not isinstance(proposal.get("reason"), str)):
        raise FilterResolutionError("Ollama filter proposal does not match schema")

    valid_dimensions = {v["code"]: v for v in variables}
    proposed = {}
    unresolved = {}
    for choice in proposal["selections"]:
        if not isinstance(choice, dict):
            raise FilterResolutionError("Malformed selected dimension")
        dim, values = choice.get("dimension"), choice.get("values")
        if (not isinstance(dim, str) or dim not in valid_dimensions
                or dim in proposed or not isinstance(values, list)
                or not values or any(not isinstance(v, str) for v in values)):
            raise FilterResolutionError(f"Invalid or duplicate selected dimension: {dim!r}")
        if any(v not in shown[dim] for v in values):
            # No accepting a model-supplied option it was not given.
            raise FilterResolutionError(f"Model invented/unlisted value for {dim!r}")
        proposed[dim] = list(dict.fromkeys(values))

    for item in proposal["unresolved"]:
        if not isinstance(item, dict):
            raise FilterResolutionError("Malformed unresolved dimension")
        dim, why = item.get("dimension"), item.get("reason")
        if (not isinstance(dim, str) or dim not in valid_dimensions
                or dim in unresolved or not isinstance(why, str)):
            raise FilterResolutionError(f"Invalid unresolved dimension: {dim!r}")
        unresolved[dim] = why or "No unambiguous choice."

    for dim in valid_dimensions:
        if dim not in proposed and dim not in unresolved:
            unresolved[dim] = "Model did not resolve this required dimension."
        if dim in proposed and dim in unresolved:
            # The conflict must not be silently ignored.
            unresolved[dim] = "Model both selected and marked this dimension unresolved."

    # A time period cannot be chosen arbitrarily just to complete a query.
    # "latest" is meaningful only if the user's normalization explicitly says so.
    time_info = (normalized or {}).get("time") or {}
    explicit_relative = time_info.get("relative")
    if not requested_years and not explicit_relative and not time_info.get("months") and not time_info.get("quarters"):
        for var in variables:
            if var.get("time") and var["code"] in proposed:
                unresolved[var["code"]] = (
                    "The question did not specify a time period; do not choose one arbitrarily."
                )

    # Do not silently accept only a subset of explicitly requested annual years.
    if requested_years:
        for var in variables:
            if not var.get("time"):
                continue
            dim = var["code"]
            if dim in proposed:
                available_years = {str(x)[:4] for x in proposed[dim] if re.match(r"^\d{4}", x)}
                if not set(requested_years).issubset(available_years):
                    unresolved[dim] = "Selected periods do not cover all requested years."
                # Monthly/quarterly observations are not automatically an annual
                # statistic. Even a valid subset is insufficient without a
                # documented aggregation strategy (sum, mean, stock, etc.).
                elif not time_info.get("months") and not time_info.get("quarters") and any(
                    re.match(r"^\d{4}(?:M\d{1,2}|Q[1-4])", str(code))
                    for code in proposed[dim]
                ):
                    unresolved[dim] = (
                        "Time selections are subannual; annual aggregation is not planned."
                    )

    public_table = {"id": table["id"], "title": table["title"], "url": table["url"]}
    if unresolved:
        return {
            "status": "needs_clarification", "table": public_table,
            "proposed_selections": proposed,
            "unresolved": [{"dimension": dim, "reason": why} for dim, why in unresolved.items()],
            "reason": proposal["reason"],
            "requires_filter_resolution": True,
        }

    try:
        prepared = prepare_query(
            table_url, proposed, catalog_path=catalog_path, max_cells=max_cells,
        )
    except PXQueryError as exc:
        return {
            "status": "needs_clarification", "table": public_table,
            "proposed_selections": proposed,
            "unresolved": [{"dimension": "(validation)", "reason": str(exc)}],
            "reason": "The proposed query did not pass PX-API metadata validation.",
            "requires_filter_resolution": True,
        }

    return {
        "status": "ready", "table": public_table,
        "selections": proposed,
        "resolved_selections": prepared["selections"],
        "cells_requested": prepared["cells_requested"],
        "reason": proposal["reason"],
        "requires_filter_resolution": False,
    }
