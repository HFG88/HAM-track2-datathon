"""Validate selections against cached Hagstofa PX-API v1 metadata and query a table.

This module does NOT interpret natural-language questions or choose a table.
Its input is an explicit table ID/URL and one or more values for EVERY dimension.
No dimension is implicitly set to 'Alls' or expanded to all possible values.

Example:
    query_table('SAM02002.px', {'Eining': 'Farþegar alls', 'Ár': '2024'})
"""

import json
import unicodedata
from functools import lru_cache
from pathlib import Path

import requests


DEFAULT_CATALOG = Path(__file__).resolve().parent / "local" / "catalog-enriched.json"
DEFAULT_MAX_CELLS = 500


class PXQueryError(ValueError):
    """A missing/ambiguous selection, invalid catalogue, or failed PX-API query."""


def _normalize(value):
    """Match human labels case- and accent-insensitively, without guessing synonyms."""
    value = unicodedata.normalize("NFKD", str(value).casefold())
    return "".join(ch for ch in value if not unicodedata.combining(ch)).strip()


@lru_cache(maxsize=4)
def load_catalog(path=DEFAULT_CATALOG):
    """Read the local snapshot once per process; pass a custom path for tests."""
    path = Path(path)
    if not path.is_file():
        raise PXQueryError(
            f"Enriched catalogue not found: {path}. "
            "Obtain local/catalog-enriched.json or run enrich_catalog.py first."
        )
    try:
        tables = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError) as exc:
        raise PXQueryError(f"Cannot read catalogue {path}: {exc}") from exc
    if not isinstance(tables, list):
        raise PXQueryError("Enriched catalogue must be a JSON array of tables")
    return tables


def _pick_one(matches, description):
    if not matches:
        raise PXQueryError(f"Unknown {description}")
    if len(matches) != 1:
        raise PXQueryError(f"Ambiguous {description}: {len(matches)} matches")
    return matches[0]


def find_table(table_ref, *, catalog_path=DEFAULT_CATALOG):
    """Look up an exact table ID or URL; reject ambiguous duplicate IDs."""
    matches = [
        table for table in load_catalog(catalog_path)
        if table.get("id") == table_ref or table.get("url") == table_ref
    ]
    table = _pick_one(matches, f"table {table_ref!r}")
    if table.get("metadata_status") != "ok":
        raise PXQueryError(f"Metadata unavailable for {table_ref!r}")
    if not isinstance(table.get("metadata", {}).get("variables"), list):
        raise PXQueryError(f"Missing dimensions for {table_ref!r}")
    return table


def _resolve_dimension(variables, name):
    exact = [var for var in variables if var.get("code") == name]
    if exact:
        return _pick_one(exact, f"dimension {name!r}")
    matches = [
        var for var in variables
        if _normalize(var.get("code", "")) == _normalize(name)
        or _normalize(var.get("text", "")) == _normalize(name)
    ]
    return _pick_one(matches, f"dimension {name!r}")


def _resolve_value(variable, requested):
    codes = variable.get("values", [])
    labels = variable.get("valueTexts", [])
    if not isinstance(codes, list) or not isinstance(labels, list) or len(codes) != len(labels):
        raise PXQueryError(f"Invalid metadata for dimension {variable.get('code')!r}")

    requested = str(requested)
    # Explicit API codes take precedence over labels. This avoids ambiguity
    # when a numeric selection could be both a code and another item's label.
    exact_codes = [i for i, code in enumerate(codes) if code == requested]
    if exact_codes:
        position = _pick_one(exact_codes, f"code {requested!r}")
    else:
        matches = [
            i for i, label in enumerate(labels)
            if _normalize(label) == _normalize(requested)
        ]
        position = _pick_one(matches, f"value {requested!r} for {variable['code']!r}")
    return {"code": codes[position], "label": labels[position]}


def prepare_query(table_ref, selections, *, catalog_path=DEFAULT_CATALOG,
                  max_cells=DEFAULT_MAX_CELLS):
    """Validate the complete selection, and build a PX-API v1 POST payload.

    selections maps dimension names/codes to a label, code, or a list of them.
    Returns a prepared request without making a network call.
    """
    if not isinstance(selections, dict):
        raise PXQueryError("Selections must be a dict of dimension -> value(s)")
    if not isinstance(max_cells, int) or max_cells < 1:
        raise PXQueryError("max_cells must be a positive integer")

    table = find_table(table_ref, catalog_path=catalog_path)
    variables = table["metadata"]["variables"]
    selections_by_code = {}

    for dimension_name, requested in selections.items():
        variable = _resolve_dimension(variables, dimension_name)
        dimension_code = variable["code"]
        if dimension_code in selections_by_code:
            raise PXQueryError(f"Dimension {dimension_code!r} was specified twice")

        values = requested if isinstance(requested, (list, tuple)) else [requested]
        if not values or any(value is None for value in values):
            raise PXQueryError(f"Empty selection for {dimension_code!r}")
        resolved = [_resolve_value(variable, value) for value in values]
        # Preserve order but don't make redundant requests for duplicate codes.
        resolved = list({value["code"]: value for value in resolved}.values())
        selections_by_code[dimension_code] = resolved

    missing = [var["code"] for var in variables if var["code"] not in selections_by_code]
    if missing:
        raise PXQueryError(
            "Specify selections for every dimension; missing: " + ", ".join(missing)
        )

    cells = 1
    for values in selections_by_code.values():
        cells *= len(values)
        if cells > max_cells:
            raise PXQueryError(
                f"Query would request {cells} cells (limit {max_cells}). "
                "Select fewer dimension values."
            )

    body = {
        "query": [
            {
                "code": var["code"],
                "selection": {
                    "filter": "item",
                    "values": [item["code"] for item in selections_by_code[var["code"]]],
                },
            }
            for var in variables
        ],
        "response": {"format": "json"},
    }
    return {
        "table": {
            "id": table["id"], "title": table["title"],
            "category": table["category"], "url": table["url"],
        },
        "selections": selections_by_code,
        "cells_requested": cells,
        "payload": body,
        "unit": table["metadata"].get("unit"),
        "source": table["metadata"].get("source"),
    }


def query_table(table_ref, selections, *, catalog_path=DEFAULT_CATALOG,
                max_cells=DEFAULT_MAX_CELLS, timeout=25, session=None):
    """POST a validated query and return the PX response with provenance.

    The `data` and `columns` fields are kept in the original PX JSON shape:
    a `values` array can represent multiple measures, not always one number.
    """
    prepared = prepare_query(
        table_ref, selections, catalog_path=catalog_path, max_cells=max_cells
    )
    http = session if session is not None else requests
    try:
        response = http.post(
            prepared["table"]["url"], json=prepared["payload"], timeout=timeout
        )
        response.raise_for_status()
        result = response.json()
    except (requests.RequestException, ValueError) as exc:
        raise PXQueryError(f"PX-API request failed for {table_ref!r}: {exc}") from exc
    if not isinstance(result, dict) or not isinstance(result.get("data"), list):
        raise PXQueryError(f"Unexpected PX-API response for {table_ref!r}: no data array")

    return {
        "table": prepared["table"],
        "source_url": prepared["table"]["url"],
        "unit": prepared["unit"],
        "source": prepared["source"],
        "selections": prepared["selections"],
        "cells_requested": prepared["cells_requested"],
        "columns": result.get("columns", []),
        "data": result["data"],
    }
