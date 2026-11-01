"""Select semantically appropriate Hagstofa PX tables from search results using Ollama.

No numerical data is fetched here. This module takes the existing
(table_dict, retrieval_score) output of table_search.search_tables and asks an
instruction-following local model to choose only from that shortlist.

The returned URLs are validated against the supplied candidates. The selection
is advisory, NOT proof the table answers the question: dimension VALUE selection
and PX-API query validation remain separate steps.
"""

import argparse
import json
import sys
from urllib.parse import unquote

import requests

DEFAULT_MODEL = "qwen3:4b-instruct"
DEFAULT_ENDPOINT = "http://localhost:11434/api/chat"


class TableSelectionError(RuntimeError):
    """An Ollama response is unusable or the model cannot be reached."""


def _summarize_table(table, index, score, question):
    """Give the LLM useful metadata without sending thousands of dimension values."""
    metadata = table.get("metadata") or {}
    dimensions = []
    # Character-level matching is only to surface helpful LABELS, not to select.
    # Do not assume that unknown labels prove the candidate unsuitable.
    question_text = str(question).casefold()
    for v in metadata.get("variables", [])[:12]:
        labels = [str(x) for x in v.get("valueTexts", [])]
        codes = [str(x) for x in v.get("values", [])]
        if v.get("time"):
            dimensions.append({
                "name": v.get("text") or v.get("code"),
                "type": "time",
                "first": labels[0] if labels else None,
                "last": labels[-1] if labels else None,
                "available_periods": len(codes),
            })
        else:
            matches = [label for label in labels if len(label) > 3 and label.casefold() in question_text][:4]
            dimensions.append({
                "name": v.get("text") or v.get("code"),
                "type": "category",
                "options": len(codes),
                "examples": labels[:3],
                "matching_labels": matches,
            })

    # Include relevant folders, which often distinguish topics with similar titles.
    url = str(table["url"])
    segments = unquote(url.split("/api/v1/", 1)[-1]).split("/")
    folder = "/".join(segments[1:-1][-3:])
    return {
        "number": index,
        "id": table.get("id"),
        "title": table.get("title"),
        "category": table.get("category"),
        "folders": folder,
        "retrieval_score": score,
        "dimensions": dimensions,
    }


def _schema(max_tables):
    return {
        "type": "object",
        "properties": {
            "selected_numbers": {
                "type": "array", "items": {"type": "integer"},
                "maxItems": max_tables,
            },
            "reason": {"type": "string", "maxLength": 300},
            "needs_clarification": {"type": "boolean"},
        },
        "required": ["selected_numbers", "reason", "needs_clarification"],
        "additionalProperties": False,
    }


def select_tables_llm(
    normalized, candidates, *, model=DEFAULT_MODEL, candidate_limit=10,
    max_tables=3, endpoint=DEFAULT_ENDPOINT, timeout=180, session=None
):
    """Ask local Ollama to choose from candidate (table, search_score) pairs.

    Selected records preserve the exact URLs provided by the catalogue.
    Returns {status, selected, reason, needs_clarification, model,
             candidates_considered, requires_filter_resolution}.
    Raises TableSelectionError on HTTP/model/invalid-output errors rather than
    silently substituting an unreliable choice.
    """
    if not isinstance(normalized, dict):
        raise TypeError("normalized must be the dictionary from normalize_question")
    if candidate_limit < 1 or max_tables < 1:
        raise ValueError("candidate_limit and max_tables must be positive")

    seen = set()
    shortlist = []
    for table, score in candidates:
        url = table.get("url")
        if not url or url in seen:
            continue
        seen.add(url)
        shortlist.append((table, score))
        if len(shortlist) == candidate_limit:
            break

    if not shortlist:
        return {
            "status": "no_candidates", "selected": [],
            "reason": "No candidate tables were retrieved.",
            "needs_clarification": True, "model": model,
            "candidates_considered": 0, "requires_filter_resolution": True,
        }

    question = normalized.get("original") or normalized.get("text") or ""
    descriptions = [
        _summarize_table(table, i, score, question)
        for i, (table, score) in enumerate(shortlist, start=1)
    ]
    system_prompt = (
        "You select official Statistics Iceland tables for a user question. "
        "Choose only candidate NUMBERS from the provided shortlist. "
        "Use meaning, population, geography, statistic type, available time "
        "periods and dimensions; do not just follow retrieval scores. "
        "For questions needing a comparison across datasets, choose several "
        "complementary tables (up to the allowed maximum). "
        "If none is suitable, return an empty selected_numbers list and "
        "needs_clarification=true. Do not invent tables, data, dimensions, "
        "numbers or statistical results. Metadata may be incomplete, "
        "so missing evidence is not always evidence of incompatibility. "
        "Table selection is not the same as selecting dimension filters. "
        "When asked about growth, use time-series figures even if growth rates "
        "must later be calculated. For comparisons, prefer complementary "
        "tables for the different subjects rather than demanding a single "
        "table containing a precomputed comparison. "
        "Give a reason of at most 35 words. Follow the supplied JSON schema."
    )
    task = {
        "question": question,
        "requested_time": normalized.get("time"),
        "requested_breakdowns": normalized.get("breakdowns"),
        "requested_measure": normalized.get("measure"),
        "intent": normalized.get("intent"),
        "maximum_tables": max_tables,
        "candidates": descriptions,
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(task, ensure_ascii=False)},
        ],
        "stream": False,
        "format": _schema(max_tables),
        "options": {"temperature": 0, "num_ctx": 8192, "num_predict": 768},
    }
    # Some small local models generate an overly long JSON `reason`. The
    # initial request already has more output tokens than the original 256;
    # if the JSON is nevertheless incomplete, retry once with a larger budget.
    http = session if session is not None else requests
    choice = None
    for attempt in range(2):
        if attempt:
            payload["options"]["num_predict"] = 1536
            payload["messages"][0]["content"] += (
                " IMPORTANT: Output ONLY the required JSON object. "
                "Keep the reason to ONE SHORT sentence."
            )
        try:
            response = http.post(endpoint, json=payload, timeout=timeout)
            response.raise_for_status()
            body = response.json()
            message = body["message"]["content"]
            choice = json.loads(message)
            break
        except requests.RequestException as exc:
            raise TableSelectionError(f"Ollama request failed: {exc}") from exc
        except (ValueError, KeyError, TypeError) as exc:
            if attempt == 0 and isinstance(exc, json.JSONDecodeError):
                print("Ollama returned malformed JSON; retrying once with a larger "
                      "response budget...", file=sys.stderr)
                continue
            done_reason = body.get("done_reason", "unknown") if isinstance(
                locals().get("body"), dict) else "unknown"
            raise TableSelectionError(
                f"Ollama returned invalid response (done_reason={done_reason}): {exc}"
            ) from exc

    if not isinstance(choice, dict):
        raise TableSelectionError("Ollama returned a non-object selection")
    indices = choice.get("selected_numbers")
    reason = choice.get("reason")
    clarify = choice.get("needs_clarification")
    if (not isinstance(indices, list) or not isinstance(reason, str)
            or type(clarify) is not bool):
        raise TableSelectionError("Ollama output does not match selection schema")
    if len(indices) > max_tables or len(set(str(i) for i in indices)) != len(indices):
        raise TableSelectionError("Ollama selected too many or duplicate candidates")
    if any(type(i) is not int or not (1 <= i <= len(shortlist)) for i in indices):
        raise TableSelectionError("Ollama selected a candidate number not in the shortlist")

    selected = []
    for number in indices:
        table, score = shortlist[number - 1]
        selected.append({
            "id": table.get("id"), "url": table["url"],
            "title": table.get("title"), "retrieval_rank": number,
            "retrieval_score": score,
        })
    return {
        "status": "selected" if selected else "no_match",
        "selected": selected, "reason": reason,
        "needs_clarification": clarify or not selected,
        "model": model, "candidates_considered": len(shortlist),
        "requires_filter_resolution": True,
    }



def main():
    parser = argparse.ArgumentParser(
        description="LLM table selection from BM25 results"
    )
    parser.add_argument("question", nargs="+", help="Icelandic statistical question")
    parser.add_argument("--mode", default="bm25", choices=("bm25", "vector", "hybrid", "cascade"))
    parser.add_argument("--candidates", type=int, default=10)
    parser.add_argument("--max-tables", type=int, default=3)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)

    args = parser.parse_args()

    from normalize import normalize_question
    from table_search import search_tables

    normalized = normalize_question(" ".join(args.question))
    results = search_tables(normalized, limit=args.candidates, mode=args.mode)

    try:
        output = select_tables_llm(
            normalized,
            results,
            candidate_limit=args.candidates,
            max_tables=args.max_tables,
            model=args.model,
            endpoint=args.endpoint,
        )
    except TableSelectionError as exc:
        parser.exit(1, f"Table selection failed: {exc}\n")

    print(json.dumps(output, ensure_ascii=False, indent=2))



if __name__ == "__main__":
    main()
