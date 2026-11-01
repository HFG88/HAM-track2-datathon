# Spyrðu Hagstofuna – Track 2 (PX-API)

**Team HAM · Gagnvist / Hagstofa hackathon 2026**

A chatbot that answers questions about Icelandic statistics in **Icelandic or English**, with **official numbers fetched live from Statistics Iceland's PX-API**, the table they came from, and a link to it.

> *Hvað bjuggu margir á Íslandi árið 2024?* → **383.726** (1. janúar 2024) · Heimild: Lykiltölur mannfjöldans 1703-2026

The language model never makes up numbers: it only reads the question, chooses the table and phrases the answer. Every number comes from the PX-API, and the page shows the exact table, the selected dimensions and the raw data behind each answer.

📄 **Visual overview (diagram, worked example, results):** [docs/prototype-overview.pdf](docs/prototype-overview.pdf)

---

## How it works

```
question (Icelandic or English)
  1. understand   Icelandic is translated to English by a fast LLM (fixes typos, puts
                  place names in base form) and gets Icelandic title keywords;
                  rules pull out years, months, places, breakdowns, intent   normalize.py, answer.py
  2. search       all 2,102 PX-API tables: BM25 on titles/variables/values
                  + vector search (qwen3-embedding), combined                 table_search.py
  3. candidates   main tables for common topics, only tables covering the
                  asked years, places found are marked                       table_choice.py
  4. choose       qwen3:32b ranks the top 10 and picks the best table         table_choice.py
  5. selections   one value per dimension: asked years/months, matching
                  places, "eftir kyni" = all values, otherwise "Alls"         answer.py
  6. fetch        validated query to the PX-API (max 500 cells)              px_query.py
  7. sum          yearly totals of monthly flows, age ranges – in Python      answer.py
  8. answer       qwen3:32b writes a short answer from the numbers only,
                  with period, unit and source                                answer.py
```

### How it connects to Statistics Iceland

- **Catalog:** `catalog.py` crawls the PX-API tree (`https://px.hagstofa.is/pxis/api/v1/is/`) and `enrich_catalog.py` fetches every table's metadata (dimensions, codes, value labels). A snapshot of all 2,102 tables is shipped in `data/` so a fresh clone works offline for search.
- **Data:** for each question, `px_query.py` validates the selection against the table's metadata and sends a **live POST** to the table's PX-API URL (`{"query": [...], "response": {"format": "json"}}`). Answers are therefore always current.

## Run it

### 1. Python
```bash
git clone https://github.com/HFG88/HAM-track2-datathon.git
cd HAM-track2-datathon
git switch prototype          # if the work is not merged into main yet
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Ollama and the models
Install Ollama from https://ollama.com, then:
```bash
ollama pull qwen3:32b              # ~20 GB – chooses the table and writes the answer
ollama pull qwen3:30b-a3b          # ~19 GB – translation and the "Hratt" mode
ollama pull qwen3-embedding:0.6b   # ~0.6 GB – vector search
```
These need a GPU machine (we used an NVIDIA GB10 with 128 GB). **On a laptop**, use small models instead (slower and less accurate, but it runs):
```bash
ollama pull qwen3:4b
export MODEL=qwen3:4b REWRITE_MODEL=qwen3:4b FAST_MODEL=qwen3:4b
```
If Ollama runs on another machine, point to it, e.g. through an SSH tunnel:
```bash
ssh -N -L 11435:localhost:11434 user@gpu-machine     # keep open
export OLLAMA_HOST=localhost:11435
```

### 3. Start
```bash
uvicorn app:app --port 8000
```
Open **http://127.0.0.1:8000/** and ask a question. On first start the catalog and the vector index are unpacked from `data/` into `local/` automatically.

From the command line:
```bash
python answer.py "Hvað búa margir í Kópavogi?"
curl -X POST http://127.0.0.1:8000/spyrja_px_api -H "Content-Type: application/json" \
     -d '{"spurning": "Hver var verðbólgan á Íslandi árið 2024?", "hamur": "venjulegt"}'
```

### Speed or quality
The page has a switch, and the API takes `"hamur"`:

| Mode | What it does | Typical time |
|---|---|---|
| `hratt` | the fast model (qwen3:30b-a3b) for every step | ~5–10 s |
| `venjulegt` (default) | fast model translates, qwen3:32b chooses and answers | ~7–25 s |
| `itarlegt` | qwen3:32b **thinks** before choosing the table and writing, 15 candidates | ~30–90 s |

## Example questions

| Question | Answer (from the PX-API) |
|---|---|
| Hvað búa margir í Kópavogi? | 40.286 (1. jan 2026) |
| hvert var ibuafjoldi hveragerðis a seinsata ari *(typos on purpose)* | 3.344 in Hveragerðisbær |
| Hver var verðbólgan á Íslandi árið 2024? | 4,8 % (12-month change, Dec 2024) |
| Hversu margir fluttu til Íslands árið 2024? | 19.789 immigrants, 15.745 emigrants |
| hvað eru margir án vinnu? | 11.900 unemployed, 5,1 % (Sept 2026) |
| Hvert er atvinnuleysið hjá konum? | 6,1 % (Q2 2026) |
| Hvað áttu margir heima á Flateyri 1990? | 396 |
| Hvað eru margir eldri en 67 ára á Íslandi? | 51.507 (ages 68+ summed) |
| Hvað komu margir ferðamenn til Íslands árið 2024? | 2.261.391 foreign passengers (12 months summed) |
| Hvað veiddist mikið af þorski 2024? | 221.677 tonnes |
| How many people live in Reykjavík? | 139,804 (1 Jan 2026) |
| Hvað búa margir á Íslandi árið 2030? | says 2030 is not available, gives the newest year |
| Hvað búa margir í Kópavogi? Hver var verðbólgan 2024? | several questions in one prompt are answered one by one |

## Project structure

```
app.py               FastAPI server: POST /spyrja_px_api, serves the frontend at /
answer.py            the pipeline: translation, selections, sums, answer, modes
normalize.py         rule-based reading of a question (years, months, places, breakdowns, intent)
question_router.py   language detection; llm_rewrite_en.py / llm_rewrite_is.py keyword rewriting
table_search.py      BM25 + vector search over the table catalog
table_choice.py      main tables, year filter, LLM ranking of the candidates
px_query.py          validated PX-API queries (every dimension checked against metadata)
catalog.py           crawls the PX-API table tree
enrich_catalog.py    fetches metadata for every table
local_data.py        unpacks data/ into local/ on first start
frontend/            the web page (HTML, CSS, JS)
data/                shipped catalog snapshot and vector index
docs/                PROTOTYPE.md, prototype-overview.pdf (diagram), LLM_REWRITE.md
eval_search.py       measures table search on test questions
AI_test/             test questions with expected tables (spurningar.txt)
tests/, test_px_query.py   unit tests:  python -m unittest discover -s tests && python -m unittest test_px_query
```

## Settings (environment variables)

| Variable | Default | |
|---|---|---|
| `OLLAMA_HOST` | `localhost:11434` | where Ollama runs |
| `MODEL` | `qwen3:32b` | table choice and answer |
| `REWRITE_MODEL` | `qwen3:30b-a3b` | translation and keywords |
| `FAST_MODEL` | `qwen3:30b-a3b` | everything in `hratt` mode |

No API keys are needed: the PX-API is open and the models run locally.

## Results

On 83 analytical test questions (`AI_test/spurningar.txt`) the right table is chosen first in 44 with qwen3:32b, against 35 for search alone. On a set of 26 everyday questions, 23 were answered correctly end to end. Details: [docs/prototype-overview.pdf](docs/prototype-overview.pdf).

## Known limitations

- Understanding and value selection are partly rule-based; unusual phrasings can pick the wrong value. The answer then usually says the data doesn't match rather than inventing a number.
- The PX-API sends no units; the unit is guessed from labels (%, index points) and otherwise left out.
- Hagur (`hagstofa-ai:latest`, Llama 3 + LoRA) was not available to us; any Ollama model can be set with `MODEL`.
- The table catalog is a snapshot; rebuild with `python catalog.py` and `python enrich_catalog.py` (then delete `local/table-vectors-*.npz` to rebuild the vectors).
