#!/usr/bin/env python3
"""Convert RepLiQA into Memory.swift eval format."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd
from huggingface_hub import hf_hub_download

from eval_data_codex_support import log, normalize_spaces, write_jsonl_atomic, write_manifest


def load_split(split_name: str) -> pd.DataFrame:
    filename = f"data/{split_name}-00000-of-00001.parquet"
    path = hf_hub_download(repo_id="ServiceNow/repliqa", repo_type="dataset", filename=filename)
    return pd.read_parquet(path)


def render_document_text(topic: str, body: str) -> str:
    topic = normalize_spaces(topic)
    body = normalize_spaces(body)
    if topic:
        return f"# {topic}\n\n{body}"
    return body


def build_manifest(splits: Sequence[str]) -> Dict[str, Any]:
    return {
        "dataset": "repliqa",
        "provenance": "external_repliqa",
        "synthetic_status": "external",
        "primary_use": "staging_candidate",
        "license_scope": "commercial_safe",
        "source_datasets": [
            {
                "name": "ServiceNow/repliqa",
                "license": "CC-BY-4.0",
                "splits": list(splits),
            }
        ],
        "known_limits": [
            "Documents are non-factual by design and well-suited to context-grounded retrieval eval.",
            "Document and query memory tags should be added with tag_eval_data_codex.py.",
        ],
        "recommended_weight": "primary",
        "promotion_status": "draft",
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert RepLiQA into Memory.swift eval format.")
    parser.add_argument("--output-dir", required=True, help="Output dataset directory.")
    parser.add_argument(
        "--splits",
        default="repliqa_0",
        help="Comma-separated RepLiQA splits to include (default: repliqa_0).",
    )
    parser.add_argument("--max-documents", type=int, default=None, help="Optional cap on converted documents.")
    parser.add_argument("--max-queries", type=int, default=None, help="Optional cap on converted queries.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    output_dir = Path(args.output_dir).resolve()
    splits = [split.strip() for split in args.splits.split(",") if split.strip()]
    if not splits:
        raise RuntimeError("At least one split is required.")

    documents: List[Dict[str, Any]] = []
    queries: List[Dict[str, Any]] = []
    seen_documents: Dict[str, str] = {}
    next_doc_id = 1
    next_query_id = 1

    for split_name in splits:
        log(f"Loading {split_name}")
        frame = load_split(split_name)
        for row in frame.to_dict(orient="records"):
            source_document_id = str(row.get("document_id", "")).strip()
            if not source_document_id:
                continue
            eval_doc_id = seen_documents.get(source_document_id)
            if eval_doc_id is None:
                if args.max_documents is not None and len(documents) >= args.max_documents:
                    continue
                eval_doc_id = f"doc-{next_doc_id:04d}"
                seen_documents[source_document_id] = eval_doc_id
                next_doc_id += 1
                documents.append(
                    {
                        "id": eval_doc_id,
                        "relative_path": str(row.get("document_path", "")).strip() or f"docs/{source_document_id}.md",
                        "kind": "markdown",
                        "text": render_document_text(
                            str(row.get("document_topic", "")),
                            str(row.get("document_extracted", "")),
                        ),
                    }
                )

            if args.max_queries is not None and len(queries) >= args.max_queries:
                continue
            query = normalize_spaces(str(row.get("question", "")))
            if len(query) < 8:
                continue
            queries.append(
                {
                    "id": f"q-{next_query_id:04d}",
                    "query": query,
                    "relevant_document_ids": [eval_doc_id],
                    "difficulty": "unknown",
                }
            )
            next_query_id += 1

    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl_atomic(output_dir / "storage_cases.jsonl", [])
    write_jsonl_atomic(output_dir / "recall_documents.jsonl", documents)
    write_jsonl_atomic(output_dir / "recall_queries.jsonl", queries)
    write_manifest(output_dir / "manifest.json", build_manifest(splits))

    log(f"Wrote {len(documents)} documents and {len(queries)} queries to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
