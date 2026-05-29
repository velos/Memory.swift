from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from .cache import (
    baseline_artifact_path,
    baseline_state_path,
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


def repo_model_path(component: str, repo_path: Path | None = None) -> Path:
    repo_path = repo_path or prepare_memory_swift_checkout()
    models_root = repo_path / "Models"
    model_map = {
        "embedding": models_root / "embedding-v1.mlpackage",
        "reranker": models_root / "reranker-v1.mlpackage",
    }
    return model_map[component]


def _write_baseline_state(component: str, state: str) -> None:
    path = baseline_state_path(component)
    ensure_dir(path.parent)
    path.write_text(state, encoding="utf-8")


def baseline_state(component: str) -> str:
    path = baseline_state_path(component)
    if not path.exists():
        return "repo"
    return path.read_text(encoding="utf-8").strip() or "repo"


def seed_baseline_model(
    component: str,
    repo_path: Path | None = None,
    source_path: Path | None = None,
    required: bool = True,
    source_state: str = "repo",
) -> Path | None:
    source = source_path or repo_model_path(component, repo_path)
    target = baseline_artifact_path(component)
    ensure_dir(target.parent)
    if target.exists():
        shutil.rmtree(target)
    if source.exists():
        shutil.copytree(source, target)
        _write_baseline_state(component, source_state)
        return target
    _write_baseline_state(component, "absent")
    if required:
        raise FileNotFoundError(f"Missing baseline CoreML artifact for {component}: {source}")
    return None


def seed_baseline_models(
    repo_path: Path | None = None,
    components: tuple[str, ...] = ("embedding", "reranker"),
    optional_components: tuple[str, ...] = ("reranker",),
) -> dict[str, Path | None]:
    repo_path = repo_path or prepare_memory_swift_checkout()
    result: dict[str, Path | None] = {}
    for component in components:
        result[component] = seed_baseline_model(
            component,
            repo_path=repo_path,
            required=component not in optional_components,
        )
    return result


def install_artifact_into_upstream(component: str, artifact_path: Path, repo_path: Path | None = None) -> Path:
    repo_path = repo_path or prepare_memory_swift_checkout()
    target = repo_model_path(component, repo_path)
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(artifact_path, target)
    return target


def restore_baseline_artifacts(
    repo_path: Path | None = None,
    components: tuple[str, ...] = ("embedding", "reranker"),
) -> None:
    repo_path = repo_path or prepare_memory_swift_checkout()
    for component in components:
        target = repo_model_path(component, repo_path)
        baseline_path = baseline_artifact_path(component)
        state = baseline_state(component)
        if state in {"absent", "generated"}:
            if target.exists():
                shutil.rmtree(target)
            continue
        if not baseline_path.exists():
            raise FileNotFoundError(f"Missing cached baseline artifact for {component}: {baseline_path}")
        restored = install_artifact_into_upstream(component, baseline_path, repo_path)
        if not restored.exists():
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
