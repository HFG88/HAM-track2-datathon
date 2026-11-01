# Metadata-validated PX filter resolution (hackathon prototype)

Four files in this package:

- `filter_resolution.py` — asks the Spark LLM for candidate **PX codes** for **every** table dimension; validates against the local enriched catalogue with `px_query.prepare_query()`. Returns `ready` or `needs_clarification` and does **not** call Statistics Iceland.
- `selection_api_v2.py` — independent FastAPI app providing `/select_table`, `/resolve_filters`, and `/answer`. The last route calls the live PX-API **only after** validation and only for a single selected table.
- `test_filter_resolution.py` — 12 offline tests (mock Ollama).
- `test_selection_api_v2.py` — 4 offline API orchestration tests (mock search/filter/query).

Copy the four files to the root of your track2 project. Do not overwrite `app.py`, `selection_api.py`, or existing teammate files.

## Offline tests

```powershell
python -m unittest -v test_filter_resolution test_selection_api_v2 test_llm_table_selection test_table_selection test_px_query
```

## Start the independent API

Keep your SSH tunnel forwarding `127.0.0.1:11435` to the Spark's Ollama `127.0.0.1:11434` open in a **different** PowerShell window.

```powershell
$env:TABLE_SELECTOR_ENDPOINT = "http://127.0.0.1:11435/api/chat"
$env:TABLE_SELECTOR_MODEL = "qwen3:30b-instruct"
python -m uvicorn selection_api_v2:app --port 8002
```

## Isolate filter resolution first (one LLM call, no PX data)

From another PowerShell in track2:

```powershell
$body = @{
    spurning = "Hversu margir farþegar komu með skemmtiferðaskipum til Reykjavíkurhafnar árið 2024?"
    table_url = "https://px.hagstofa.is/pxis/api/v1/is/Atvinnuvegir/ferdathjonusta/farthegar/1_farthegar/SAM02002.px"
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://127.0.0.1:8002/resolve_filters" -Method Post `
    -ContentType "application/json; charset=utf-8" `
    -Body ([System.Text.Encoding]::UTF8.GetBytes($body)) |
    ConvertTo-Json -Depth 10
```

Expected when correctly resolved: `status: ready`, `Eining: ["1"]`, `Ár: ["2024"]`, and `cells_requested: 1`. These are expected outputs **not** previously observed outputs of the newly written resolver.

## End-to-end prototype (two LLM calls + live PX query)

```powershell
$body = @{
    spurning = "Hversu margir farþegar komu með skemmtiferðaskipum til Reykjavíkurhafnar árið 2024?"
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://127.0.0.1:8002/answer" -Method Post `
    -ContentType "application/json; charset=utf-8" `
    -Body ([System.Text.Encoding]::UTF8.GetBytes($body)) |
    ConvertTo-Json -Depth 12
```

On success `status` is `data_retrieved`; the raw result is under `px_result.data`, and `source_url` is provided. The already-verified manual result for the cruise-passenger table was `321966` for year 2024; the new automatic endpoint has **not** yet been live-tested.

## Deliberate limits

- A `ready` result means **every dimension and selected code is valid**, **not** that the model understood every statistical nuance correctly. Check the label and meaning.
- No automatic calculation of year-over-year changes, growth rates, percentage changes, weighted averages, or cross-table comparisons.
- No arbitrary fallback for missing time periods or dimensions; large option lists are selectively exposed and may require clarification.
- No auto-aggregation from monthly to annual observations.
- The default code still uses BM25 for retrieval and the normalizer still misclassifies certain Icelandic comparisons; this package does not change teammates' components.
