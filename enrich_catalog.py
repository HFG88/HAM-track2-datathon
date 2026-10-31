"""Enrich a locally crawled Statistics Iceland PxWeb table catalogue.

Reads local/catalog-full.json, caches each table's complete GET metadata response,
and builds local/catalog-enriched.json. No numerical observations are downloaded.

Usage (from the project root):
    python enrich_catalog.py --limit 10
    python enrich_catalog.py

Interrupted and repeated runs reuse successfully downloaded metadata.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import requests

CATALOG_PATH = Path("local/catalog-full.json")
CACHE_DIR = Path("local/catalog-metadata")
OUTPUT_PATH = Path("local/catalog-enriched.json")
FAILURES_PATH = Path("local/catalog-metadata-failures.json")
RETRY_STATUSES = {429, 500, 502, 503, 504}


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json_atomic(path: Path, data) -> None:
    """Replace a JSON file only after its complete new contents are written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def metadata_path(table: dict) -> Path:
    # Full URL, not table ID: codes may repeat in different catalogue branches.
    digest = hashlib.sha256(table["url"].encode("utf-8")).hexdigest()
    return CACHE_DIR / f"{digest}.json"


def valid_metadata(metadata) -> bool:
    return (
        isinstance(metadata, dict)
        and isinstance(metadata.get("variables"), list)
        and len(metadata["variables"]) > 0
    )


def cached_metadata(table: dict):
    path = metadata_path(table)
    if not path.is_file():
        return None
    try:
        payload = read_json(path)
        if valid_metadata(payload):
            return payload
    except (ValueError, OSError):
        pass
    return None


def retry_delay(response, attempt: int) -> float:
    retry_after = response.headers.get("Retry-After", "")
    if retry_after.isdecimal():
        return min(float(retry_after), 60.0)
    return min(float(2 ** (attempt - 1)), 30.0)


def fetch_metadata(
    session: requests.Session,
    url: str,
    delay: float,
    max_attempts: int,
) -> dict:
    """Retrieve metadata with modest pacing and retries for transient failures."""
    for attempt in range(1, max_attempts + 1):
        response = None
        try:
            response = session.get(
                url,
                headers={"Accept": "application/json"},
                timeout=(10, 30),
            )
            if response.status_code in RETRY_STATUSES:
                if attempt < max_attempts:
                    pause = retry_delay(response, attempt)
                    print(f"  HTTP {response.status_code}; retrying in {pause:g}s")
                    time.sleep(pause)
                    continue
            response.raise_for_status()
            metadata = response.json()
            if not valid_metadata(metadata):
                raise ValueError("Metadata response has no usable 'variables' list")
            return metadata
        except (requests.Timeout, requests.ConnectionError) as exc:
            if attempt == max_attempts:
                raise RuntimeError(f"Network error after {max_attempts} attempts: {exc}") from exc
            pause = min(float(2 ** (attempt - 1)), 30.0)
            print(f"  Connection error; retrying in {pause:g}s")
            time.sleep(pause)
        finally:
            # Only slow actual HTTP calls, not reads from the local cache.
            time.sleep(delay)
    raise RuntimeError(f"Request failed after {max_attempts} attempts (HTTP {response.status_code if response is not None else 'unknown'})")


def build_enriched_catalog(tables: list[dict]) -> tuple[int, int]:
    """Stream the aggregate JSON file to avoid holding all metadata in RAM."""
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT_PATH.with_name(OUTPUT_PATH.name + ".tmp")
    with_metadata = 0
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("[\n")
        for index, table in enumerate(tables):
            metadata = cached_metadata(table)
            if metadata is not None:
                with_metadata += 1
            record = {
                **table,
                "metadata_status": "ok" if metadata is not None else "missing",
                "metadata": metadata,  # Includes variables: code, text, values, valueTexts, time, ...
            }
            if index:
                handle.write(",\n")
            handle.write("  ")
            json.dump(record, handle, ensure_ascii=False, separators=(",", ":"))
        handle.write("\n]\n")
    os.replace(temporary, OUTPUT_PATH)
    return with_metadata, len(tables) - with_metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="Maximum NEW metadata requests this run")
    parser.add_argument("--delay", type=float, default=0.5, help="Seconds between HTTP requests (default 0.5)")
    parser.add_argument("--attempts", type=int, default=3, help="Attempts for transient failures (default 3)")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 0:
        parser.error("--limit must be nonnegative")
    if args.delay < 0 or args.attempts < 1:
        parser.error("--delay must be >= 0 and --attempts >= 1")

    if not CATALOG_PATH.is_file():
        parser.error(f"Missing {CATALOG_PATH}. Run the full catalogue crawler first.")
    tables = read_json(CATALOG_PATH)
    if not isinstance(tables, list) or not all(
        isinstance(t, dict) and isinstance(t.get("url"), str) and t.get("url", "").startswith("https://px.hagstofa.is/")
        for t in tables
    ):
        parser.error("Catalog must be a list of tables with official https://px.hagstofa.is/ URLs")
    if len({t["url"] for t in tables}) != len(tables):
        parser.error("Duplicate table URLs found; fix the source catalogue first")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Catalogue: {len(tables)} tables | Cache: {CACHE_DIR}")
    downloaded = 0
    reused = 0
    failed = []
    interrupted = False

    try:
        with requests.Session() as session:
            for index, table in enumerate(tables, start=1):
                if cached_metadata(table) is not None:
                    reused += 1
                    continue
                if args.limit is not None and downloaded >= args.limit:
                    break
                downloaded += 1  # Counts attempted NEW requests, including failures.
                code = table.get("id", "(unknown)")
                print(f"[{index}/{len(tables)}] {code}", flush=True)
                try:
                    metadata = fetch_metadata(session, table["url"], args.delay, args.attempts)
                    save_json_atomic(metadata_path(table), metadata)
                except (requests.RequestException, RuntimeError, ValueError, OSError) as exc:
                    print(f"  FAILED: {exc}", flush=True)
                    failed.append({"id": code, "url": table["url"], "error": str(exc)})
    except KeyboardInterrupt:
        interrupted = True
        print("\nInterrupted. Cached successes will be preserved.")

    # Build a usable partial aggregate even if --limit was used or Ctrl+C was pressed.
    ok, missing = build_enriched_catalog(tables)
    save_json_atomic(FAILURES_PATH, failed)
    print(f"\nCached from earlier runs: {reused}")
    print(f"New requests attempted: {downloaded}")
    print(f"Failed this run: {len(failed)}")
    print(f"Enriched: {ok}/{len(tables)} | Missing: {missing}")
    print(f"Saved: {OUTPUT_PATH} ({OUTPUT_PATH.stat().st_size / 1024 / 1024:.1f} MiB)")
    print(f"Failures recorded: {FAILURES_PATH}")
    if missing:
        print("Run 'python enrich_catalog.py' again to fetch the remaining metadata.")
    elif interrupted:
        print("Metadata is complete despite interruption.")


if __name__ == "__main__":
    main()
