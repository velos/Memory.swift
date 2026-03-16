#!/usr/bin/env python3
"""Merge audit results and review queues into a single triage report."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from eval_data_codex_support import load_jsonl


def normalized_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    result = []
    for item in value:
        cleaned = str(item).strip()
        if cleaned and cleaned not in result:
            result.append(cleaned)
    return result


def audit_summary(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "verdict": str(row.get("verdict", "")).strip().lower() or "unknown",
        "confidence": str(row.get("confidence", "")).strip().lower() or "low",
        "issues": normalized_list(row.get("issues")),
        "notes": str(row.get("notes", "")).strip(),
        "suggested_memory_type": row.get("suggested_memory_type"),
        "suggested_memory_types": normalized_list(row.get("suggested_memory_types")),
        "suggested_difficulty": row.get("suggested_difficulty"),
        "suggested_required_spans": normalized_list(row.get("suggested_required_spans")),
        "suggested_relevant_document_ids": normalized_list(row.get("suggested_relevant_document_ids")),
        "backend": row.get("backend"),
        "model": row.get("model"),
        "audited_at": row.get("audited_at"),
    }


def compare_suggestions(left: Dict[str, Any], right: Dict[str, Any]) -> List[str]:
    differences: List[str] = []
    for key in (
        "verdict",
        "suggested_memory_type",
        "suggested_difficulty",
    ):
        if left.get(key) != right.get(key):
            differences.append(key)
    for key in (
        "suggested_memory_types",
        "suggested_required_spans",
        "suggested_relevant_document_ids",
    ):
        if normalized_list(left.get(key)) != normalized_list(right.get(key)):
            differences.append(key)
    return differences


def make_priority(
    *,
    review_context: Optional[Dict[str, Any]],
    opencode_row: Optional[Dict[str, Any]],
    minimax_row: Optional[Dict[str, Any]],
    disagreements: List[str],
) -> str:
    if review_context:
        return "p1"
    if disagreements:
        return "p1"
    for row in (opencode_row, minimax_row):
        if row and str(row.get("verdict", "")).strip().lower() == "reject":
            return "p1"
    for row in (opencode_row, minimax_row):
        if row and str(row.get("verdict", "")).strip().lower() == "needs_edit":
            return "p2"
    return "p3"


def build_triage_row(
    packet_row: Dict[str, Any],
    *,
    review_context: Optional[Dict[str, Any]],
    opencode_row: Optional[Dict[str, Any]],
    minimax_row: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    op_summary = audit_summary(opencode_row) if opencode_row else None
    mini_summary = audit_summary(minimax_row) if minimax_row else None
    disagreements = compare_suggestions(op_summary or {}, mini_summary or {}) if op_summary and mini_summary else []

    needs_human = bool(review_context) or bool(disagreements)
    for row in (op_summary, mini_summary):
        if not row:
            continue
        if row["verdict"] in {"needs_edit", "reject"}:
            needs_human = True
            break

    if not needs_human:
        return None

    return {
        "packet_id": packet_row.get("packet_id"),
        "dataset": packet_row.get("dataset"),
        "entry_type": packet_row.get("entry_type"),
        "sample_reason": packet_row.get("sample_reason"),
        "priority": make_priority(
            review_context=review_context,
            opencode_row=op_summary,
            minimax_row=mini_summary,
            disagreements=disagreements,
        ),
        "review_queue_reason": review_context.get("reason") if review_context else None,
        "opencode": op_summary,
        "minimax": mini_summary,
        "disagreements": disagreements,
        "query": packet_row.get("query"),
        "current_memory_types": packet_row.get("current_memory_types"),
        "current_difficulty": packet_row.get("current_difficulty"),
        "current_memory_type": packet_row.get("current_memory_type"),
        "required_spans": packet_row.get("required_spans"),
        "source_id": packet_row.get("source_id"),
        "relative_path": packet_row.get("relative_path"),
        "source_document_id": packet_row.get("source_document_id"),
    }


def render_markdown(rows: Sequence[Dict[str, Any]], dataset_name: str, generated_at: str) -> str:
    lines = [
        f"# Audit Triage: {dataset_name}",
        "",
        f"- Generated: {generated_at}",
        f"- Total rows needing review: {len(rows)}",
        "",
    ]

    for row in rows:
        lines.append(f"## {row['packet_id']} `{row['priority']}`")
        lines.append("")
        lines.append(f"- Type: `{row.get('entry_type')}`")
        lines.append(f"- Sample reason: `{row.get('sample_reason')}`")
        if row.get("review_queue_reason"):
            lines.append(f"- Review queue reason: `{row.get('review_queue_reason')}`")
        if row.get("disagreements"):
            lines.append(f"- Backend disagreements: `{row.get('disagreements')}`")
        if row.get("query"):
            lines.append(f"- Query: {row.get('query')}")
        if row.get("current_memory_type"):
            lines.append(f"- Current memory type: `{row.get('current_memory_type')}`")
        if row.get("current_memory_types"):
            lines.append(f"- Current memory types: `{row.get('current_memory_types')}`")
        if row.get("current_difficulty"):
            lines.append(f"- Current difficulty: `{row.get('current_difficulty')}`")
        if row.get("required_spans"):
            lines.append(f"- Required spans: `{row.get('required_spans')}`")
        lines.append("")
        if row.get("opencode"):
            lines.append("### OpenCode")
            lines.append("")
            lines.append(f"- Verdict: `{row['opencode'].get('verdict')}` `{row['opencode'].get('confidence')}`")
            if row["opencode"].get("issues"):
                lines.append(f"- Issues: `{row['opencode'].get('issues')}`")
            if row["opencode"].get("notes"):
                lines.append(f"- Notes: {row['opencode'].get('notes')}")
            lines.append("")
        if row.get("minimax"):
            lines.append("### MiniMax")
            lines.append("")
            lines.append(f"- Verdict: `{row['minimax'].get('verdict')}` `{row['minimax'].get('confidence')}`")
            if row["minimax"].get("issues"):
                lines.append(f"- Issues: `{row['minimax'].get('issues')}`")
            if row["minimax"].get("notes"):
                lines.append(f"- Notes: {row['minimax'].get('notes')}")
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge audit outputs into a triage report.")
    parser.add_argument("--packet", required=True, help="Path to packet.jsonl.")
    parser.add_argument("--review-queue", default=None, help="Optional review_queue.jsonl override.")
    parser.add_argument("--opencode", default=None, help="Optional audit_results.opencode.jsonl path.")
    parser.add_argument("--minimax", default=None, help="Optional audit_results.minimax.jsonl path.")
    parser.add_argument("--output-prefix", default=None, help="Output prefix (defaults beside packet).")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    packet_path = Path(args.packet).resolve()
    dataset_dir = packet_path.parent
    dataset_name = dataset_dir.name
    review_queue_path = Path(args.review_queue).resolve() if args.review_queue else packet_path.parents[1] / dataset_name / "review_queue.jsonl"
    opencode_path = Path(args.opencode).resolve() if args.opencode else dataset_dir / "audit_results.opencode.jsonl"
    minimax_path = Path(args.minimax).resolve() if args.minimax else dataset_dir / "audit_results.minimax.jsonl"
    output_prefix = Path(args.output_prefix).resolve() if args.output_prefix else dataset_dir / "triage"

    packet_rows = load_jsonl(packet_path)
    opencode_rows = load_jsonl(opencode_path)
    minimax_rows = load_jsonl(minimax_path)
    review_rows = load_jsonl(review_queue_path)

    review_by_packet: Dict[str, Dict[str, Any]] = {}
    review_by_record: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for row in review_rows:
        review_by_record[(str(row.get("mode", "")), str(row.get("record_id", "")))] = row

    for packet_row in packet_rows:
        entry_type = str(packet_row.get("entry_type", ""))
        source_id = str(packet_row.get("source_id", ""))
        packet_id = str(packet_row.get("packet_id", ""))
        if entry_type == "query":
            review = review_by_record.get(("query-tags", source_id))
        elif entry_type == "document":
            review = review_by_record.get(("document-tags", source_id))
        else:
            review = review_by_record.get(("storage-cases", source_id))
            if not review and packet_row.get("source_document_id"):
                review = review_by_record.get(("storage-cases", str(packet_row.get("source_document_id"))))
        if review:
            review_by_packet[packet_id] = review

    opencode_by_packet = {str(row.get("packet_id", "")): row for row in opencode_rows}
    minimax_by_packet = {str(row.get("packet_id", "")): row for row in minimax_rows}

    triage_rows: List[Dict[str, Any]] = []
    followup_packet_rows: List[Dict[str, Any]] = []
    for packet_row in packet_rows:
        packet_id = str(packet_row.get("packet_id", ""))
        triage = build_triage_row(
            packet_row,
            review_context=review_by_packet.get(packet_id),
            opencode_row=opencode_by_packet.get(packet_id),
            minimax_row=minimax_by_packet.get(packet_id),
        )
        if triage:
            triage_rows.append(triage)
            followup_packet_rows.append(packet_row)

    triage_rows.sort(key=lambda row: (row.get("priority", "p9"), str(row.get("entry_type", "")), str(row.get("source_id", ""))))
    generated_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    (output_prefix.with_suffix(".jsonl")).write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in triage_rows),
        encoding="utf-8",
    )
    (output_prefix.with_suffix(".md")).write_text(render_markdown(triage_rows, dataset_name, generated_at), encoding="utf-8")
    (output_prefix.with_suffix(".summary.json")).write_text(
        json.dumps(
            {
                "dataset": dataset_name,
                "generated_at": generated_at,
                "counts": {
                    "packet_rows": len(packet_rows),
                    "opencode_rows": len(opencode_rows),
                    "minimax_rows": len(minimax_rows),
                    "review_queue_rows": len(review_rows),
                    "triage_rows": len(triage_rows),
                    "followup_packet_rows": len(followup_packet_rows),
                },
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    (output_prefix.parent / "followup_packet.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in followup_packet_rows),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
