"""End-to-end smoke test against a running compose stack.

Creates a workspace, uploads a PDF, waits for ingestion, runs every retrieval mode, re-uploads to check dedupe, and
deletes the workspace. Usage: python scripts/smoke.py <path-to.pdf> "<query>" [--api http://127.0.0.1:18000]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import httpx


def wait_ready(client: httpx.Client, workspace: str, timeout_s: float) -> dict[str, object]:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        documents = client.get(f"/api/v1/workspaces/{workspace}/documents").raise_for_status().json()
        version = documents[0]["versions"][0]
        if version["ingestion_status"] in ("ready", "failed"):
            return dict(version)
        time.sleep(2)
    raise TimeoutError("ingestion did not finish in time")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf", type=Path)
    parser.add_argument("query")
    parser.add_argument("--api", default="http://127.0.0.1:18000")
    parser.add_argument("--timeout", type=float, default=600.0)
    args = parser.parse_args()

    raw = args.pdf.read_bytes()
    with httpx.Client(base_url=args.api, timeout=300.0) as client:
        print("health:", client.get("/health").json())  # noqa: T201
        workspace = client.post("/api/v1/workspaces", json={"name": "smoke"}).raise_for_status().json()["id"]
        try:
            first = client.post(
                f"/api/v1/workspaces/{workspace}/documents", files={"file": (args.pdf.name, raw, "application/pdf")}
            ).raise_for_status()
            print("upload:", first.status_code, first.json()["status"])  # noqa: T201
            started = time.monotonic()
            version = wait_ready(client, workspace, args.timeout)
            print("ingested in seconds:", round(time.monotonic() - started, 1), version["ingestion_status"])  # noqa: T201
            if version["ingestion_status"] != "ready":
                print("FAILED:", version["error_code"], version["error_message"])  # noqa: T201
                return 1
            print("extractor:", version["extractor"])  # noqa: T201

            for mode in ("bm25", "dense", "hybrid_rrf", "hybrid_rrf_rerank"):
                result = (
                    client.post(
                        f"/api/v1/workspaces/{workspace}/search", json={"query": args.query, "mode": mode, "k": 3}
                    )
                    .raise_for_status()
                    .json()
                )
                if not result["evidence"]:
                    print("FAILED: no evidence for", mode)  # noqa: T201
                    return 1
                top = result["evidence"][0]
                print(  # noqa: T201
                    json.dumps(
                        {
                            "mode": mode,
                            "page": top["page_start"],
                            "section": top["section_path"],
                            "offsets": [top["char_start"], top["char_end"]],
                            "total_ms": round(result["timings_ms"]["total"], 1),
                            "text": top["text"][:90],
                        }
                    )
                )

            again = client.post(
                f"/api/v1/workspaces/{workspace}/documents", files={"file": (args.pdf.name, raw, "application/pdf")}
            ).raise_for_status()
            print("re-upload:", again.status_code, again.json()["status"])  # noqa: T201
            if again.json()["status"] != "duplicate":
                return 1
        finally:
            print("delete:", client.delete(f"/api/v1/workspaces/{workspace}").status_code)  # noqa: T201
    return 0


if __name__ == "__main__":
    sys.exit(main())
