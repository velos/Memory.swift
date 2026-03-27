#!/usr/bin/env python3
"""Convert MultiDoc2Dial into Memory.swift eval format."""

from __future__ import annotations

import argparse
import json
import os
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from eval_data_codex_support import log, normalize_spaces, write_jsonl_atomic, write_manifest

MULTIDOC2DIAL_URL = "https://doc2dial.github.io/multidoc2dial/file/multidoc2dial.zip"


def download_and_extract(cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    zip_path = cache_dir / "multidoc2dial.zip"
    extracted_root = cache_dir / "multidoc2dial_extracted"
    marker = extracted_root / "multidoc2dial" / "multidoc2dial_doc.json"
    if marker.exists():
        return extracted_root

    log(f"Downloading {MULTIDOC2DIAL_URL}")
    urllib.request.urlretrieve(MULTIDOC2DIAL_URL, zip_path)
    if extracted_root.exists():
        for child in extracted_root.iterdir():
            if child.is_dir():
                for nested in sorted(child.rglob("*"), reverse=True):
                    if nested.is_file():
                        nested.unlink()
                    else:
                        nested.rmdir()
                child.rmdir()
            else:
                child.unlink()
    extracted_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(extracted_root)
    return extracted_root


def load_json(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def render_document_text(title: str, body: str) -> str:
    title = normalize_spaces(title)
    body = normalize_spaces(body)
    if title:
        return f"# {title}\n\n{body}"
    return body


def iter_dialogue_files(root: Path, splits: Sequence[str]) -> Iterable[Path]:
    for split in splits:
        path = root / "multidoc2dial" / f"multidoc2dial_dial_{split}.json"
        if path.exists():
            yield path


def convert_documents(doc_payload: Dict[str, Any], max_documents: Optional[int]) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    records: List[Dict[str, Any]] = []
    raw_to_eval_id: Dict[str, str] = {}
    count = 0
    doc_data = doc_payload.get("doc_data", {})
    if not isinstance(doc_data, dict):
        raise ValueError("MultiDoc2Dial doc_data is missing or invalid.")

    for domain in sorted(doc_data):
        domain_docs = doc_data.get(domain, {})
        if not isinstance(domain_docs, dict):
            continue
        for raw_doc_id in sorted(domain_docs):
            source = domain_docs.get(raw_doc_id, {})
            if not isinstance(source, dict):
                continue
            eval_doc_id = f"doc-{count + 1:04d}"
            raw_to_eval_id[str(raw_doc_id)] = eval_doc_id
            title = str(source.get("title", "")).strip()
            body = str(source.get("doc_text", "")).strip()
            records.append(
                {
                    "id": eval_doc_id,
                    "relative_path": f"{sanitize_component(domain)}/{sanitize_component(str(raw_doc_id))}.md",
                    "kind": "markdown",
                    "text": render_document_text(title, body),
                }
            )
            count += 1
            if max_documents is not None and count >= max_documents:
                return records, raw_to_eval_id
    return records, raw_to_eval_id


def sanitize_component(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "-" for ch in value)
    cleaned = cleaned.strip("-")
    return cleaned or "item"


def convert_queries(
    dialogue_paths: Sequence[Path],
    raw_to_eval_id: Dict[str, str],
    max_queries: Optional[int],
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    seen = set()
    next_id = 1
    for path in dialogue_paths:
        payload = load_json(path)
        dial_data = payload.get("dial_data", {})
        if not isinstance(dial_data, dict):
            continue
        for domain in sorted(dial_data):
            dialogues = dial_data.get(domain, [])
            if not isinstance(dialogues, list):
                continue
            for dialogue in dialogues:
                if not isinstance(dialogue, dict):
                    continue
                turns = dialogue.get("turns", [])
                if not isinstance(turns, list):
                    continue
                for turn in turns:
                    if not isinstance(turn, dict):
                        continue
                    if str(turn.get("role", "")).lower() != "user":
                        continue
                    references = turn.get("references", [])
                    if not isinstance(references, list):
                        continue
                    relevant: List[str] = []
                    for reference in references:
                        if not isinstance(reference, dict):
                            continue
                        raw_doc_id = str(reference.get("doc_id", "")).strip()
                        mapped = raw_to_eval_id.get(raw_doc_id)
                        if mapped and mapped not in relevant:
                            relevant.append(mapped)
                    query = normalize_spaces(str(turn.get("utterance", "")))
                    if len(query) < 8 or not relevant:
                        continue
                    dedupe_key = (query.lower(), tuple(relevant))
                    if dedupe_key in seen:
                        continue
                    seen.add(dedupe_key)
                    records.append(
                        {
                            "id": f"q-{next_id:04d}",
                            "query": query,
                            "relevant_document_ids": relevant[:3],
                            "difficulty": "unknown",
                        }
                    )
                    next_id += 1
                    if max_queries is not None and len(records) >= max_queries:
                        return records
    return records


def build_manifest(source_splits: Sequence[str]) -> Dict[str, Any]:
    return {
        "dataset": "multidoc2dial",
        "provenance": "external_multidoc2dial",
        "synthetic_status": "external",
        "primary_use": "staging_candidate",
        "license_scope": "commercial_safe",
        "source_datasets": [
            {
                "name": "IBM/multidoc2dial",
                "license": "Apache-2.0",
                "splits": list(source_splits),
            }
        ],
        "known_limits": [
            "Converted from a public multi-document dialogue benchmark.",
            "Queries are derived from user turns with document references.",
            "Document and query memory tags should be added with tag_eval_data_codex.py.",
        ],
        "recommended_weight": "primary",
        "promotion_status": "draft",
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert MultiDoc2Dial into Memory.swift eval format.")
    parser.add_argument("--output-dir", required=True, help="Output dataset directory.")
    parser.add_argument(
        "--splits",
        default="train,validation",
        help="Comma-separated dialogue splits to include (default: train,validation).",
    )
    parser.add_argument("--cache-dir", default="/tmp/multidoc2dial_cache", help="Download/extract cache dir.")
    parser.add_argument("--max-documents", type=int, default=None, help="Optional cap on converted documents.")
    parser.add_argument("--max-queries", type=int, default=None, help="Optional cap on converted queries.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    output_dir = Path(args.output_dir).resolve()
    cache_dir = Path(args.cache_dir).resolve()
    splits = [split.strip() for split in args.splits.split(",") if split.strip()]
    if not splits:
        raise RuntimeError("At least one split is required.")

    root = download_and_extract(cache_dir)
    doc_payload = load_json(root / "multidoc2dial" / "multidoc2dial_doc.json")
    documents, raw_to_eval_id = convert_documents(doc_payload, args.max_documents)
    queries = convert_queries(list(iter_dialogue_files(root, splits)), raw_to_eval_id, args.max_queries)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl_atomic(output_dir / "storage_cases.jsonl", [])
    write_jsonl_atomic(output_dir / "recall_documents.jsonl", documents)
    write_jsonl_atomic(output_dir / "recall_queries.jsonl", queries)
    write_manifest(output_dir / "manifest.json", build_manifest(splits))

    log(f"Wrote {len(documents)} documents and {len(queries)} queries to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
