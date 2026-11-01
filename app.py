"""FastAPI server for the Hagstofa chatbot.

POST /spyrja_px_api {"spurning": "..."} runs the whole pipeline in answer.py:
question -> table search -> LLM table choice -> PX-API data -> LLM answer.
The frontend is served at http://127.0.0.1:8000/

Run:
    OLLAMA_HOST=localhost:11435 uvicorn app:app --reload
"""

import threading
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from answer import answer_question

FRONTEND = Path(__file__).parent / "frontend"

app = FastAPI(title="Hagstofa Snjallt API Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# One question at a time talks to Ollama, so parallel requests don't make the GPU
# swap or run out of memory (the "ollama_lock" from the Track 2 blueprint).
ollama_lock = threading.Lock()


class SpurningRequest(BaseModel):
    spurning: str


@app.post("/spyrja_px_api")
def spyrja_px_api(req: SpurningRequest):
    with ollama_lock:
        result = answer_question(req.spurning)

    table = result.get("table") or {}
    return {
        # Fields the frontend already uses.
        "svar": result["answer"],
        "töflu_slóð": table.get("url"),
        # Extra detail for showing how the answer was found.
        "tafla": table,
        "víddir": result.get("selections"),
        "athugasemdir": result.get("notes", []),
        "gögn": result.get("data", [])[:20],
        "tungumál": result.get("language"),
        "tími": result.get("timings"),
        "villa": result.get("error"),
    }


@app.get("/")
def frontend():
    return FileResponse(FRONTEND / "index.html")


app.mount("/", StaticFiles(directory=FRONTEND), name="frontend")
