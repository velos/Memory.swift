#!/usr/bin/env python3
"""Shared helpers for Codex-backed eval data generation and tagging."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

DEFAULT_MODEL = "gpt-5.2"


def log(message: str) -> None:
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def truncate_for_log(value: str, limit: int = 220) -> str:
    compact = normalize_spaces(value)
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


def render_jsonl(records: Sequence[Dict[str, Any]]) -> str:
    return "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + ("\n" if records else "")


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []

    result: List[Dict[str, Any]] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {path}:{line_number}: {exc}") from exc
        if not isinstance(parsed, dict):
            raise ValueError(f"Expected JSON object in {path}:{line_number}")
        result.append(parsed)
    return result


def write_jsonl_atomic(path: Path, records: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.parent / f".{path.name}.tmp"
    temp_path.write_text(render_jsonl(records), encoding="utf-8")
    temp_path.replace(path)


def load_manifest(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}

    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON manifest at {path}: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"Expected manifest object in {path}")
    return parsed


def write_manifest(path: Path, manifest: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def extract_json_payload(raw: str) -> Any:
    raw = raw.strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    fence_match = re.search(r"```(?:json)?\s*(.*?)```", raw, flags=re.DOTALL | re.IGNORECASE)
    if fence_match:
        fenced = fence_match.group(1).strip()
        try:
            return json.loads(fenced)
        except json.JSONDecodeError:
            pass

    object_match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if object_match:
        candidate = object_match.group(0).strip()
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    array_match = re.search(r"\[.*\]", raw, flags=re.DOTALL)
    if array_match:
        candidate = array_match.group(0).strip()
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    raise ValueError("Model output did not contain valid JSON.")


def ensure_json_object_with_list(payload: Any, key: str) -> List[Dict[str, Any]]:
    if isinstance(payload, dict):
        items = payload.get(key)
    else:
        items = payload

    if not isinstance(items, list):
        raise ValueError(f"Expected list field '{key}' in model output.")

    records: List[Dict[str, Any]] = []
    for item in items:
        if isinstance(item, dict):
            records.append(item)
    return records


def summarize_codex_failure(stdout: str, stderr: str, return_code: int) -> str:
    stderr_clean = stderr.strip()
    stdout_clean = stdout.strip()
    lines = [line.strip() for line in (stderr_clean + "\n" + stdout_clean).splitlines() if line.strip()]

    for line in reversed(lines):
        if line.startswith("ERROR:"):
            return line
    for line in reversed(lines):
        if "invalid_json_schema" in line:
            return line
    for line in reversed(lines):
        if "mcp startup" in line.lower():
            return line

    if lines:
        return lines[-1]
    return f"codex exited with code {return_code}"


class CodexClient:
    def __init__(
        self,
        *,
        codex_bin: str,
        model: str,
        workspace: Path,
        timeout_seconds: int,
        max_retries_per_request: int,
        config_overrides: Sequence[str],
    ) -> None:
        resolved = shutil.which(codex_bin)
        if not resolved:
            raise RuntimeError(f"Could not find codex binary '{codex_bin}' on PATH.")

        self.codex_bin = resolved
        self.model = model
        self.workspace = workspace
        self.timeout_seconds = timeout_seconds
        self.max_retries_per_request = max_retries_per_request
        self.config_overrides = list(config_overrides)

    def ensure_login(self) -> None:
        command = [self.codex_bin, "login", "status"]
        for override in self.config_overrides:
            command.extend(["-c", override])
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            error = truncate_for_log(result.stderr.strip() or result.stdout.strip() or "login status failed")
            raise RuntimeError(
                "Codex login check failed. Run `codex login` and retry. "
                f"Details: {error}"
            )

    def create_message(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        output_schema: Dict[str, Any],
        progress_label: Optional[str] = None,
    ) -> str:
        final_prompt = (
            "You are generating or tagging evaluation data.\n"
            "Follow the output schema and return JSON only.\n"
            "Do not run shell commands.\n\n"
            f"SYSTEM:\n{system_prompt}\n\n"
            f"TASK:\n{user_prompt}\n"
        )

        last_error: Optional[str] = None
        for attempt in range(1, self.max_retries_per_request + 1):
            label = progress_label or "codex request"
            started_at = time.time()
            log(f"{label}: request {attempt}/{self.max_retries_per_request}")

            with tempfile.TemporaryDirectory(prefix="codex_eval_") as tmpdir:
                output_path = Path(tmpdir) / "last_message.txt"
                schema_path = Path(tmpdir) / "output_schema.json"
                schema_path.write_text(json.dumps(output_schema), encoding="utf-8")

                command = [
                    self.codex_bin,
                    "exec",
                    "-C",
                    str(self.workspace),
                    "--sandbox",
                    "read-only",
                    "-m",
                    self.model,
                ]

                for override in self.config_overrides:
                    command.extend(["-c", override])

                command.extend(
                    [
                        "--output-schema",
                        str(schema_path),
                        "--output-last-message",
                        str(output_path),
                        final_prompt,
                    ]
                )

                try:
                    result = subprocess.run(
                        command,
                        capture_output=True,
                        text=True,
                        timeout=self.timeout_seconds,
                    )
                except subprocess.TimeoutExpired:
                    last_error = f"timeout after {self.timeout_seconds}s"
                else:
                    if result.returncode != 0:
                        last_error = summarize_codex_failure(result.stdout, result.stderr, result.returncode)
                    else:
                        try:
                            response = output_path.read_text(encoding="utf-8").strip()
                        except Exception as exc:
                            last_error = f"failed reading codex output: {exc}"
                        else:
                            if response:
                                elapsed = time.time() - started_at
                                log(f"{label}: response in {elapsed:.1f}s")
                                return response
                            last_error = "codex produced empty output"

            if attempt < self.max_retries_per_request:
                sleep_seconds = min(12.0, 1.8 ** attempt)
                log(
                    f"{label}: failed ({truncate_for_log(last_error or 'unknown error')}); "
                    f"retrying in {sleep_seconds:.1f}s"
                )
                time.sleep(sleep_seconds)
            else:
                log(f"{label}: failed ({truncate_for_log(last_error or 'unknown error')})")

        raise RuntimeError(f"Codex request failed after retries: {last_error}")
