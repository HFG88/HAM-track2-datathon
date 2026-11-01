# Prototype: the whole pipeline

Visual overview with a diagram, a worked example and results: [prototype-overview.pdf](prototype-overview.pdf).

A working end-to-end chatbot. Not perfect, but it answers real questions with real numbers from Hagstofa, with the source.

```
question
  → answer.py             Icelandic: fast LLM translates to English (fixes typos, place names)
                          and gives Icelandic title keywords; English: question_router.py
  → normalize.py          years, months, breakdowns, keywords
  → table_search.py       hybrid search (BM25 + vectors), top candidates
  → table_choice.py       main tables for common topics + year filter, LLM ranks top 3
  → answer.py             builds selections for every dimension
  → px_query.py           validated PX-API query, real data
  → answer.py             LLM writes a short answer with numbers, period and source
  → app.py                POST /spyrja_px_api, frontend at /
```

## Run it

You need three things: the Python packages, the catalog files in `local/`, and Ollama with the models.

**1. Python**
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**2. Catalog files:** nothing to do. The catalog snapshot and the vector index are shipped in `data/` and unpacked into `local/` on first start (`local_data.py`). To rebuild from the PX-API: `python catalog.py`, then `python enrich_catalog.py`.

**3. Ollama with `qwen3:32b`, `qwen3:30b-a3b` and `qwen3-embedding:0.6b`.** We run them on the GB10 machine and reach it through an SSH tunnel:
```bash
ssh -N -L 11435:localhost:11434 hackathon@<machine-ip>    # leave open
```

**4. Start the server** (new terminal, in the venv):
```bash
OLLAMA_HOST=localhost:11435 uvicorn app:app --port 8000
```
Open http://127.0.0.1:8000/ and ask a question.

Or from the command line:
```bash
OLLAMA_HOST=localhost:11435 python answer.py "Hvað búa margir í Kópavogi?"
curl -X POST http://127.0.0.1:8000/spyrja_px_api -H "Content-Type: application/json" \
     -d '{"spurning": "Hver var verðbólgan á Íslandi árið 2024?"}'
```

## Settings (environment variables)

| Variable | Default | What |
|---|---|---|
| `OLLAMA_HOST` | `localhost:11434` | where Ollama runs |
| `MODEL` | `qwen3:32b` | table choice and answer |
| `REWRITE_MODEL` | `qwen3:30b-a3b` | translation + keywords (fast) |
| `TABLE_MODEL`, `ANSWER_MODEL` | `MODEL` | override one step |

## API response

```json
{
  "svar": "Á Íslandi bjuggu 383.726 á 1. janúar árið 2024. Heimild: …",
  "töflu_slóð": "https://px.hagstofa.is/…/MAN00000.px",
  "tafla": {"id": "MAN00000.px", "title": "Lykiltölur mannfjöldans 1703-2026", "url": "…"},
  "víddir": "Ár: 2024; Eining: Mannfjöldi 1. janúar",
  "athugasemdir": [],
  "gögn": ["Ár: 2024 | Eining: Mannfjöldi 1. janúar → … = 383726"],
  "tungumál": "is",
  "tími": {"table_choice": 4.0, "px_query": 0.2, "answer": 5.3, "total": 9.5},
  "villa": null
}
```

## Tested questions (qwen3:32b on the GB10)

| Question | Table | Answer |
|---|---|---|
| Hvað búa margir í Kópavogi? | MAN02005 | 40.286 (2026) |
| Hvað bjuggu margir á Íslandi árið 2024? | MAN00000 | 383.726 (1. jan 2024) |
| Hver var verðbólgan á Íslandi árið 2024? | VIS01000 | 4,8% (ársbreyting, des 2024) |
| Hversu margir fluttu til Íslands árið 2024? | MAN01400 | 19.789 aðfluttir, 15.745 brottfluttir |
| Hvert er atvinnuleysið? | VIN01002 | 4,3% (2025) |
| Hver er launavísitalan? | LAU04000 | 1157,7 (sept 2026) |
| How many people live in Reykjavík? | MAN03270 | 139,804 (1 Jan 2026) |
| Hvað búa margir á Íslandi árið 2030? | MAN00000 | says 2030 isn't available, gives 2026 |
| asdkj qwpe | – | "Ég skildi ekki spurninguna…" |

Answers take about 7–30 seconds. Icelandic questions are first translated to English by the fast model (under 1 s), because the English route understood questions much better (typos, inflected place names like "Hveragerðis", "síðasta ári").

## Known limitations

- **Selections are rule-based.** If a question's words don't match a table's value labels, the total ("Alls") is used. The answer then says the place wasn't found, rather than inventing a number.
- **Monthly tables with only a year asked** use the year's last month (e.g. December). That's right for stocks (population, index level, annual inflation) but not for flows (passengers per year would need the sum of all months).
- **Main tables** (`MAIN_TABLES` in `table_choice.py`) are hand-picked for common topics. Other topics depend on the search + LLM.
- **Occasionally** the LLM returns the data rows instead of an answer; then the data is shown as-is with the source.
- **Hagur** (`hagstofa-ai:latest`) is not used, because we don't have it. Any Ollama model can be set with `MODEL`.
