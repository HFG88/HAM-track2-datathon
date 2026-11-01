"""Isolated end-to-end prototype: select table, resolve filters, optionally query PX.

Run, without modifying the team's original app.py or selection_api.py:
    python -m uvicorn selection_api_v2:app --port 8002

Uses TABLE_SELECTOR_ENDPOINT / TABLE_SELECTOR_MODEL (Spark via SSH tunnel).
Outputs raw sourced PX rows, not unverified natural-language conclusions.
"""

import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from normalize import QuestionError, normalize_question
from table_search import search_tables
from llm_table_selection import TableSelectionError, select_tables_llm
from filter_resolution import FilterResolutionError, resolve_filters
from px_query import PXQueryError, query_table

app = FastAPI(title="Hagstofa Verified Query Prototype")


class QuestionRequest(BaseModel):
    spurning: str = Field(min_length=2)


class FilterRequest(QuestionRequest):
    table_url: str = Field(min_length=5)


def model_options():
    return {
        "model": os.getenv("TABLE_SELECTOR_MODEL", "qwen3:30b-instruct"),
        "endpoint": os.getenv("TABLE_SELECTOR_ENDPOINT", "http://127.0.0.1:11435/api/chat"),
    }


@app.post("/select_table")
def select_table(req: QuestionRequest):
    try:
        normalized = normalize_question(req.spurning)
    except QuestionError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    candidates = search_tables(normalized, limit=20, mode="bm25")
    if not candidates:
        return {"status": "no_candidates", "question": req.spurning, "selection": None}
    try:
        selection = select_tables_llm(
            normalized, candidates, candidate_limit=20, max_tables=3,
            **model_options(),
        )
    except TableSelectionError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return {
        "status": "selection_only", "question": req.spurning,
        "intent": normalized.get("intent"), "selection": selection,
        "warning": "No statistical data has been queried.",
    }


@app.post("/resolve_filters")
def resolve_filters_endpoint(req: FilterRequest):
    """Resolve for an explicit URL; makes no PX-API network request."""
    try:
        normalized = normalize_question(req.spurning)
        return resolve_filters(
            req.spurning, req.table_url, normalized=normalized,
            **model_options(),
        )
    except QuestionError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except PXQueryError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except FilterResolutionError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.post("/answer")
def answer(req: QuestionRequest):
    """End-to-end for one table only; no hidden defaults or computed statistics."""
    selected_response = select_table(req)
    selection = selected_response.get("selection")
    if not selection or selection.get("status") != "selected":
        return {"status": "needs_clarification", "selection": selection,
                "warning": "No usable table selected. No PX query executed."}
    choices = selection["selected"]
    if len(choices) != 1:
        return {
            "status": "needs_multi_table_planning", "selected_tables": choices,
            "warning": "Multiple tables require comparison/aggregation planning; no PX query executed.",
        }
    table_url = choices[0]["url"]
    filters = resolve_filters_endpoint(FilterRequest(spurning=req.spurning, table_url=table_url))
    if filters["status"] != "ready":
        return {"status": "needs_clarification", "table": choices[0],
                "filters": filters, "warning": "No PX query executed."}
    try:
        data = query_table(table_url, filters["selections"])
    except PXQueryError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return {
        "status": "data_retrieved", "question": req.spurning,
        "table_selection": choices[0], "filters": filters,
        "px_result": data,
        "warning": "Raw PX observations only; no derived calculations or units inferred.",
    }
