#!/usr/bin/env python3
"""Scan runtime and test code for benchmark-derived answer phrases.

This guard is intentionally conservative: it looks for exact benchmark answer
surfaces, benchmark IDs/names in runtime code, and rescue-oriented wording that
usually means a focused slice leaked into product behavior. Broad domain aliases
such as RSU/equity incentive or Cantonese/Yue embroidery are not flagged.
"""

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path


DEFAULT_PATHS = [
    Path("Sources/Memory"),
    Path("Tests/MemoryTests"),
]

DEFAULT_EXCLUDES = {
    ".build",
    ".git",
    "Evals",
    "Explorations",
    "Models",
    "references",
}


@dataclass(frozen=True)
class Rule:
    name: str
    pattern: re.Pattern[str]
    severity: str
    rationale: str


def phrase_rule(phrase: str, *, severity: str = "error", rationale: str = "benchmark answer phrase") -> Rule:
    escaped = re.escape(phrase)
    pattern = re.compile(rf"(?<![A-Za-z0-9_]){escaped}(?![A-Za-z0-9_])", re.IGNORECASE)
    return Rule(name=phrase, pattern=pattern, severity=severity, rationale=rationale)


RULES: list[Rule] = [
    # External benchmark/data answer surfaces that should not appear in runtime
    # retrieval/extraction heuristics or tests that assert heuristic behavior.
    phrase_rule("john mulaney"),
    phrase_rule("kid gorgeous"),
    phrase_rule("hasan minhaj"),
    phrase_rule("homecoming king"),
    phrase_rule("ali wong"),
    phrase_rule("mike birbiglia"),
    phrase_rule("trader joe"),
    phrase_rule("trader joe's"),
    phrase_rule("walmart"),
    phrase_rule("thrive market"),
    phrase_rule("marketing coordinator"),
    phrase_rule("senior marketing specialist"),
    phrase_rule("harvard"),
    phrase_rule("zumba"),
    phrase_rule("bodypump"),
    phrase_rule("hip hop abs"),
    phrase_rule("portland"),
    phrase_rule("afi fest"),
    phrase_rule("apple tv+"),
    phrase_rule("disney+"),
    phrase_rule("hulu"),
    phrase_rule("for all mankind"),
    phrase_rule("emma"),
    phrase_rule("rachel lee"),
    phrase_rule("plankchallenge"),
    phrase_rule("#plankchallenge"),
    phrase_rule("eastern sierra"),
    phrase_rule("yosemite"),
    phrase_rule("big sur"),
    phrase_rule("monterey"),
    phrase_rule("austin"),
    phrase_rule("toaster oven"),
    phrase_rule("coffee maker"),
    phrase_rule("teacher wu"),
    phrase_rule("he meiling"),
    phrase_rule("pine crane"),
    phrase_rule("yu xiaowei"),
    # Phrases that are broad in isolation but were previously used as exact
    # benchmark rescues. Keep the guard scoped to multi-token answer surfaces.
    phrase_rule("road trip"),
    phrase_rule("camping trip"),
    phrase_rule("solo trip"),
    phrase_rule("chocolate cake"),
    phrase_rule("caramel ganache"),
    # Benchmark identifiers should stay in eval tooling/docs, not shipped
    # Memory runtime behavior.
    phrase_rule("lifebench", rationale="benchmark identifier in scanned code"),
    phrase_rule("personamem", rationale="benchmark identifier in scanned code"),
    phrase_rule("agent-memory-benchmark", rationale="benchmark identifier in scanned code"),
    phrase_rule("present a poster", rationale="benchmark-specific query trigger"),
    Rule(
        name="longmemeval runtime mention",
        pattern=re.compile(r"(?<![A-Za-z0-9_])longmemeval(?![A-Za-z0-9_])", re.IGNORECASE),
        severity="error",
        rationale="benchmark identifier in scanned code",
    ),
    Rule(
        name="benchmark query/document id",
        pattern=re.compile(r"\b(?:q-gpt4_[0-9a-f]+|q-[0-9a-f]{8}|doc-answer_[A-Za-z0-9_]+)\b"),
        severity="error",
        rationale="benchmark case/document ID",
    ),
    Rule(
        name="rescue wording",
        pattern=re.compile(r"\brescue(?:[-_ ]?(?:term|phrase|case|hit|query|slice|benchmark|heuristic)s?)?\b", re.IGNORECASE),
        severity="warning",
        rationale="rescue wording should not describe runtime behavior",
    ),
]


def iter_files(paths: list[Path], excludes: set[str]) -> list[Path]:
    files: list[Path] = []
    for root in paths:
        if root.is_file():
            files.append(root)
            continue
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if any(part in excludes for part in path.parts):
                continue
            if path.suffix.lower() not in {".swift", ".py", ".md", ".json", ".jsonl"}:
                continue
            files.append(path)
    return sorted(files)


def line_for_offset(text: str, offset: int) -> tuple[int, str]:
    line_no = text.count("\n", 0, offset) + 1
    line_start = text.rfind("\n", 0, offset) + 1
    line_end = text.find("\n", offset)
    if line_end == -1:
        line_end = len(text)
    return line_no, text[line_start:line_end].strip()


def is_allowlisted(path: Path, line: str, allowlist_patterns: list[re.Pattern[str]]) -> bool:
    haystack = f"{path}:{line}"
    return any(pattern.search(haystack) for pattern in allowlist_patterns)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path, help="Paths to scan. Defaults to runtime/test code.")
    parser.add_argument("--allow", action="append", default=[], help="Regex allowlist matched against '<path>:<line>'.")
    parser.add_argument("--strict-warnings", action="store_true", help="Treat warnings as failures.")
    parser.add_argument("--json", type=Path, help="Optional JSON report path.")
    args = parser.parse_args()

    paths = args.paths or DEFAULT_PATHS
    allowlist_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in args.allow]
    findings: list[dict[str, object]] = []

    for path in iter_files(paths, DEFAULT_EXCLUDES):
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for rule in RULES:
            for match in rule.pattern.finditer(text):
                line_no, line = line_for_offset(text, match.start())
                if is_allowlisted(path, line, allowlist_patterns):
                    continue
                findings.append(
                    {
                        "path": str(path),
                        "line": line_no,
                        "severity": rule.severity,
                        "rule": rule.name,
                        "rationale": rule.rationale,
                        "text": line,
                    }
                )

    if args.json:
        import json

        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({"findings": findings}, indent=2) + "\n", encoding="utf-8")

    errors = [finding for finding in findings if finding["severity"] == "error"]
    warnings = [finding for finding in findings if finding["severity"] == "warning"]

    for finding in findings:
        print(
            f"[{finding['severity']}] {finding['path']}:{finding['line']} "
            f"{finding['rule']} - {finding['rationale']}: {finding['text']}"
        )

    if not findings:
        print("[benchmark-leakage] no benchmark-derived phrases found.")
    else:
        print(f"[benchmark-leakage] {len(errors)} error(s), {len(warnings)} warning(s).")

    if errors or (args.strict_warnings and warnings):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
