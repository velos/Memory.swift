#!/usr/bin/env python3
"""Convert QASPER into Memory.swift eval format."""

from __future__ import annotations

import argparse
import io
import json
import tarfile
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from eval_data_codex_support import log, normalize_spaces, write_jsonl_atomic, write_manifest

QASPER_TRAIN_DEV_URL = "https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-train-dev-v0.3.tgz"
QASPER_TEST_URL = "https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-test-and-evaluator-v0.3.tgz"
QASPER_FILES = {
    "train": "qasper-train-v0.3.json",
    "dev": "qasper-dev-v0.3.json",
    "test": "qasper-test-v0.3.json",
}


def download_file(url: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        return destination
    log(f"Downloading {url}")
    urllib.request.urlretrieve(url, destination)
    return destination


def load_split_payload(archive_path: Path, filename: str) -> Dict[str, Any]:
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive.getmembers():
            if member.name.endswith(filename):
                fileobj = archive.extractfile(member)
                if fileobj is None:
                    break
                data = json.load(io.TextIOWrapper(fileobj, encoding="utf-8"))
                if not isinstance(data, dict):
                    raise ValueError(f"Expected JSON object in {filename}")
                return data
    raise FileNotFoundError(f"Could not find {filename} in {archive_path}")


def render_document_text(paper: Dict[str, Any]) -> str:
    title = normalize_spaces(str(paper.get("title", "")))
    abstract = normalize_spaces(str(paper.get("abstract", "")))
    sections = paper.get("full_text", [])
    lines: List[str] = []
    if title:
        lines.append(f"# {title}")
    if abstract:
        lines.extend(["", "## Abstract", "", abstract])
    if isinstance(sections, list):
        for section in sections:
            if not isinstance(section, dict):
                continue
            name = normalize_spaces(str(section.get("section_name", "")))
            paragraphs = section.get("paragraphs", [])
            rendered_paragraphs = [
                normalize_spaces(str(paragraph))
                for paragraph in paragraphs
                if isinstance(paragraph, str) and normalize_spaces(paragraph)
            ]
            if name:
                lines.extend(["", f"## {name}"])
            if rendered_paragraphs:
                lines.extend([""] + rendered_paragraphs)
    return "\n".join(lines).strip()


def has_answerable_content(answers: Any) -> bool:
    if not isinstance(answers, list):
        return False
    for answer_wrapper in answers:
        if not isinstance(answer_wrapper, dict):
            continue
        answer = answer_wrapper.get("answer", {})
        if not isinstance(answer, dict):
            continue
        if bool(answer.get("unanswerable")):
            continue
        if answer.get("extractive_spans") or answer.get("free_form_answer") or answer.get("evidence"):
            return True
    return False


def build_manifest(splits: Sequence[str]) -> Dict[str, Any]:
    return {
        "dataset": "qasper",
        "provenance": "external_qasper",
        "synthetic_status": "external",
        "primary_use": "staging_candidate",
        "license_scope": "commercial_safe",
        "source_datasets": [
            {
                "name": "allenai/qasper",
                "license": "CC-BY-4.0",
                "splits": list(splits),
            }
        ],
        "known_limits": [
            "Queries are paper-level information-seeking questions from a scientific QA benchmark.",
            "Converted documents are entire papers, so retrieval difficulty is partly long-context driven.",
            "Document and query memory tags should be added with tag_eval_data_codex.py.",
        ],
        "recommended_weight": "secondary",
        "promotion_status": "draft",
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert QASPER into Memory.swift eval format.")
    parser.add_argument("--output-dir", required=True, help="Output dataset directory.")
    parser.add_argument(
        "--splits",
        default="train,dev",
        help="Comma-separated QASPER splits to include (default: train,dev).",
    )
    parser.add_argument("--cache-dir", default="/tmp/qasper_cache", help="Download cache directory.")
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

    train_dev_archive = download_file(QASPER_TRAIN_DEV_URL, cache_dir / "qasper-train-dev-v0.3.tgz")
    test_archive = download_file(QASPER_TEST_URL, cache_dir / "qasper-test-and-evaluator-v0.3.tgz")

    documents: List[Dict[str, Any]] = []
    queries: List[Dict[str, Any]] = []
    seen_documents: Dict[str, str] = {}
    next_doc_id = 1
    next_query_id = 1

    for split in splits:
        archive_path = test_archive if split == "test" else train_dev_archive
        payload = load_split_payload(archive_path, QASPER_FILES[split])
        for paper_id, paper in payload.items():
            if not isinstance(paper, dict):
                continue
            eval_doc_id = seen_documents.get(paper_id)
            if eval_doc_id is None:
                if args.max_documents is not None and len(documents) >= args.max_documents:
                    continue
                eval_doc_id = f"doc-{next_doc_id:04d}"
                next_doc_id += 1
                seen_documents[paper_id] = eval_doc_id
                documents.append(
                    {
                        "id": eval_doc_id,
                        "relative_path": f"papers/{paper_id}.md",
                        "kind": "markdown",
                        "text": render_document_text(paper),
                    }
                )

            qas = paper.get("qas", [])
            if not isinstance(qas, list):
                continue
            for qa in qas:
                if not isinstance(qa, dict):
                    continue
                if args.max_queries is not None and len(queries) >= args.max_queries:
                    continue
                if not has_answerable_content(qa.get("answers")):
                    continue
                question = normalize_spaces(str(qa.get("question", "")))
                if len(question) < 8:
                    continue
                queries.append(
                    {
                        "id": f"q-{next_query_id:04d}",
                        "query": question,
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
