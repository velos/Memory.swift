#!/usr/bin/env python3
"""Shared helpers for eval data audit packet generation and model-assisted review."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, TypeVar

DEFAULT_OPENCODE_MODEL = "opencode/nemotron-3-super-free"

T = TypeVar("T")


def truncate_text(text: str, limit: int) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


def highlight_spans(text: str, spans: Sequence[str]) -> str:
    highlighted = text
    unique_spans = [span for span in spans if span]
    for span in sorted(unique_spans, key=len, reverse=True):
        highlighted = highlighted.replace(span, f"[[{span}]]")
    return highlighted


def parse_numeric_tail(value: str) -> Optional[int]:
    match = re.search(r"(\d+)(?!.*\d)", value)
    if not match:
        return None
    return int(match.group(1))


def stable_order_key(seed: int, *parts: str) -> str:
    joined = "|".join([str(seed), *parts])
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()


def stratified_sample(
    records: Sequence[T],
    target: int,
    *,
    key_fn: Callable[[T], Tuple[str, ...]],
    id_fn: Callable[[T], str],
    seed: int,
) -> List[T]:
    if target <= 0 or not records:
        return []

    buckets: Dict[Tuple[str, ...], List[T]] = defaultdict(list)
    for record in records:
        buckets[key_fn(record)].append(record)

    ordered_buckets: List[Tuple[Tuple[str, ...], List[T]]] = []
    for bucket_key, bucket_records in sorted(buckets.items(), key=lambda item: item[0]):
        ordered = sorted(
            bucket_records,
            key=lambda record: stable_order_key(seed, *bucket_key, id_fn(record)),
        )
        ordered_buckets.append((bucket_key, ordered))

    selected: List[T] = []
    bucket_indexes = {bucket_key: 0 for bucket_key, _ in ordered_buckets}
    while len(selected) < min(target, len(records)):
        made_progress = False
        for bucket_key, bucket_records in ordered_buckets:
            index = bucket_indexes[bucket_key]
            if index >= len(bucket_records):
                continue
            selected.append(bucket_records[index])
            bucket_indexes[bucket_key] = index + 1
            made_progress = True
            if len(selected) >= min(target, len(records)):
                break
        if not made_progress:
            break
    return selected


def extract_opencode_text(stdout: str) -> str:
    texts: List[str] = []
    for raw_line in stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if payload.get("type") != "text":
            continue
        part = payload.get("part")
        if isinstance(part, dict):
            text = str(part.get("text", "")).strip()
            if text:
                texts.append(text)
    if texts:
        return "\n".join(texts).strip()
    return stdout.strip()


def summarize_opencode_failure(stdout: str, stderr: str, return_code: int) -> str:
    stderr_clean = stderr.strip()
    stdout_clean = stdout.strip()
    lines = [line.strip() for line in (stderr_clean + "\n" + stdout_clean).splitlines() if line.strip()]
    if lines:
        return lines[-1]
    return f"opencode exited with code {return_code}"


class OpenCodeClient:
    def __init__(
        self,
        *,
        opencode_bin: str,
        model: str,
        workspace: Path,
        timeout_seconds: int,
        xdg_data_home: Optional[Path] = None,
        xdg_config_home: Optional[Path] = None,
        xdg_cache_home: Optional[Path] = None,
    ) -> None:
        resolved = shutil.which(opencode_bin)
        if not resolved:
            raise RuntimeError(f"Could not find opencode binary '{opencode_bin}' on PATH.")
        self.opencode_bin = resolved
        self.model = model
        self.workspace = workspace
        self.timeout_seconds = timeout_seconds
        self.xdg_data_home = xdg_data_home or workspace / ".opencode-data"
        self.xdg_config_home = xdg_config_home or workspace / ".opencode-config"
        self.xdg_cache_home = xdg_cache_home or workspace / ".opencode-cache"

    def create_message(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        progress_label: Optional[str] = None,
    ) -> str:
        final_prompt = (
            "You are auditing evaluation data.\n"
            "Return JSON only.\n"
            "Do not use markdown fences.\n\n"
            f"SYSTEM:\n{system_prompt}\n\n"
            f"TASK:\n{user_prompt}\n"
        )
        env = os.environ.copy()
        env["XDG_DATA_HOME"] = str(self.xdg_data_home)
        env["XDG_CONFIG_HOME"] = str(self.xdg_config_home)
        env["XDG_CACHE_HOME"] = str(self.xdg_cache_home)
        self.xdg_data_home.mkdir(parents=True, exist_ok=True)
        self.xdg_config_home.mkdir(parents=True, exist_ok=True)
        self.xdg_cache_home.mkdir(parents=True, exist_ok=True)

        command = [
            self.opencode_bin,
            "--model",
            self.model,
            "run",
            "--format",
            "json",
            "--dir",
            str(self.workspace),
            final_prompt,
        ]
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=self.timeout_seconds,
            env=env,
        )
        if result.returncode != 0:
            raise RuntimeError(summarize_opencode_failure(result.stdout, result.stderr, result.returncode))

        response = extract_opencode_text(result.stdout)
        if not response:
            label = progress_label or "opencode request"
            raise RuntimeError(f"{label} produced empty output")
        return response


def split_batches_by_kind(items: Sequence[Dict[str, Any]], batch_size: int) -> List[List[Dict[str, Any]]]:
    batches: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = []
    current_kind: Optional[str] = None
    for item in items:
        item_kind = str(item.get("entry_type", "")).strip()
        if not current:
            current = [item]
            current_kind = item_kind
            continue
        if item_kind != current_kind or len(current) >= batch_size:
            batches.append(current)
            current = [item]
            current_kind = item_kind
            continue
        current.append(item)
    if current:
        batches.append(current)
    return batches

