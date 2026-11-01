"""Unpack the shipped data in data/ into local/ so a fresh clone works without crawling.

data/catalog-enriched.json.gz          all 2,102 PX-API tables with their metadata
data/table-vectors-…-title.npz         qwen3-embedding vectors of the table titles

local/ is what the code reads (and is not in git). Rebuild the data from scratch with
catalog.py + enrich_catalog.py; the vectors are rebuilt automatically if missing.
"""

import gzip
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).parent
DATA = ROOT / "data"
LOCAL = ROOT / "local"


def ensure_local_data():
    """Create the files in local/ from data/ if they are missing. Safe to call every start."""
    LOCAL.mkdir(exist_ok=True)

    enriched = LOCAL / "catalog-enriched.json"
    packed = DATA / "catalog-enriched.json.gz"
    if not enriched.exists() and packed.exists():
        with gzip.open(packed, "rb") as source, open(enriched, "wb") as target:
            shutil.copyfileobj(source, target)

    # The plain catalog (titles and urls only) is used for the normaliser's word list.
    full = LOCAL / "catalog-full.json"
    if not full.exists() and enriched.exists():
        tables = json.loads(enriched.read_text(encoding="utf-8"))
        slim = [{k: t[k] for k in ("id", "title", "url", "category")} for t in tables]
        full.write_text(json.dumps(slim, ensure_ascii=False, indent=2), encoding="utf-8")

    for vectors in DATA.glob("table-vectors-*.npz"):
        if not (LOCAL / vectors.name).exists():
            shutil.copy(vectors, LOCAL / vectors.name)


if __name__ == "__main__":
    ensure_local_data()
    print("local/ is ready:", sorted(p.name for p in LOCAL.iterdir()))
