from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

from .config import CACHE_NAMESPACE, MEMORY_SWIFT_REPO_ROOT


def cache_root() -> Path:
    return Path.home() / ".cache" / CACHE_NAMESPACE


def repo_cache_key() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=MEMORY_SWIFT_REPO_ROOT,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "workspace"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def upstream_root() -> Path:
    return ensure_dir(cache_root() / "workspace" / repo_cache_key())


def datasets_root() -> Path:
    return ensure_dir(cache_root() / "datasets" / repo_cache_key())


def artifacts_root() -> Path:
    return ensure_dir(cache_root() / "artifacts")


def baselines_root() -> Path:
    return ensure_dir(artifacts_root() / "baselines")


def candidates_root() -> Path:
    return ensure_dir(artifacts_root() / "candidates")


def runs_root() -> Path:
    return ensure_dir(cache_root() / "runs")


def tokenizer_root() -> Path:
    return ensure_dir(cache_root() / "tokenizers")


def hardware_profile_path() -> Path:
    return ensure_dir(cache_root() / "hardware") / "hardware_profile.json"


def memory_swift_repo_path() -> Path:
    return MEMORY_SWIFT_REPO_ROOT


def memory_swift_build_path() -> Path:
    return MEMORY_SWIFT_REPO_ROOT / ".build" / "release" / "memory_eval"


def baseline_artifact_path(component: str) -> Path:
    return baselines_root() / component / "current.mlpackage"


def candidate_artifact_path(component: str) -> Path:
    return candidates_root() / component / "candidate.mlpackage"


def checkpoint_path(component: str) -> Path:
    return candidates_root() / component / "candidate_weights.npz"


def metrics_path(component: str) -> Path:
    return runs_root() / f"{component}_last_metrics.json"


@dataclass
class RunManifest:
    component: str
    artifact_path: str
    checkpoint_path: str
    metrics_path: str


def write_manifest(path: Path, manifest: RunManifest) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(asdict(manifest), indent=2), encoding="utf-8")
