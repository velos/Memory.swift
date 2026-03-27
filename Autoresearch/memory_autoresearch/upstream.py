from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from .cache import (
    baseline_artifact_path,
    ensure_dir,
    memory_swift_build_path,
    memory_swift_repo_path,
)
from .config import DEFAULT_MEMORY_EVAL_PROFILE


def _run(command: list[str], cwd: Path | None = None) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def prepare_memory_swift_checkout() -> Path:
    repo_path = memory_swift_repo_path()
    if not (repo_path / "Package.swift").exists():
        raise FileNotFoundError(f"Memory.swift repo root not found at {repo_path}")
    return repo_path


def build_memory_eval_binary(repo_path: Path | None = None) -> Path:
    repo_path = repo_path or prepare_memory_swift_checkout()
    _run(["swift", "build", "-c", "release", "--product", "memory_eval"], cwd=repo_path)
    return memory_swift_build_path()


def upstream_evals_root(repo_path: Path | None = None) -> Path:
    repo_path = repo_path or prepare_memory_swift_checkout()
    return repo_path / "Evals"


def seed_baseline_models(repo_path: Path | None = None) -> dict[str, Path]:
    repo_path = repo_path or prepare_memory_swift_checkout()
    models_root = repo_path / "Models"
    baseline_map = {
        "embedding": models_root / "embedding-v1.mlpackage",
        "reranker": models_root / "reranker-v1.mlpackage",
    }
    result = {}
    for component, source in baseline_map.items():
        target = baseline_artifact_path(component)
        ensure_dir(target.parent)
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(source, target)
        result[component] = target
    return result


def install_artifact_into_upstream(component: str, artifact_path: Path, repo_path: Path | None = None) -> Path:
    repo_path = repo_path or prepare_memory_swift_checkout()
    models_root = repo_path / "Models"
    target_map = {
        "embedding": models_root / "embedding-v1.mlpackage",
        "reranker": models_root / "reranker-v1.mlpackage",
    }
    target = target_map[component]
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(artifact_path, target)
    return target


def restore_baseline_artifacts(repo_path: Path | None = None) -> None:
    repo_path = repo_path or prepare_memory_swift_checkout()
    for component in ("embedding", "reranker"):
        target = install_artifact_into_upstream(component, baseline_artifact_path(component), repo_path)
        if not target.exists():
            raise FileNotFoundError(f"Failed to restore baseline artifact for {component}")


def run_memory_eval(
    dataset_root: Path,
    output_path: Path,
    repo_path: Path | None = None,
    profile: str = DEFAULT_MEMORY_EVAL_PROFILE,
) -> dict:
    repo_path = repo_path or prepare_memory_swift_checkout()
    binary = memory_swift_build_path()
    if not binary.exists():
        binary = build_memory_eval_binary(repo_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(binary),
        "run",
        "--profile",
        profile,
        "--dataset-root",
        str(dataset_root),
        "--output",
        str(output_path),
    ]
    _run(command, cwd=repo_path)
    return json.loads(output_path.read_text(encoding="utf-8"))
