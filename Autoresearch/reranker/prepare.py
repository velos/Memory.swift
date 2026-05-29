"""Fixed bootstrapper for the Memory.swift reranker autoresearch loop."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

AUTORESEARCH_ROOT = Path(__file__).resolve().parents[1]
if str(AUTORESEARCH_ROOT) not in sys.path:
    sys.path.insert(0, str(AUTORESEARCH_ROOT))

from memory_autoresearch.cache import (
    baseline_artifact_path,
    baseline_state_path,
    configure_setup,
    datasets_root,
    hardware_profile_path,
    memory_swift_repo_path,
)
from memory_autoresearch.data import materialize_dataset_cache
from memory_autoresearch.hardware import load_or_create_profile
from memory_autoresearch.upstream import (
    build_memory_eval_binary,
    prepare_memory_swift_checkout,
    repo_model_path,
    seed_baseline_model,
    upstream_evals_root,
)

SETUP_NAME = "reranker"
SETUP_ROOT = Path(__file__).resolve().parent
configure_setup(SETUP_NAME)


def _seed_reranker_baseline(repo_path: Path) -> tuple[Path, str]:
    repo_source = repo_model_path("reranker", repo_path)
    if repo_source.exists():
        target = seed_baseline_model("reranker", repo_path=repo_path, source_state="repo")
        if target is None:
            raise FileNotFoundError(f"Failed to seed reranker baseline from {repo_source}")
        return target, "repo"

    target = baseline_artifact_path("reranker")
    if not target.exists():
        converter = repo_path / "Scripts" / "convert_tinybert_reranker_coreml.py"
        subprocess.run(
            [sys.executable, str(converter), "--output-path", str(target)],
            cwd=repo_path,
            check=True,
        )
    state_path = baseline_state_path("reranker")
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text("generated", encoding="utf-8")
    return target, "generated"


def main() -> None:
    repo_path = prepare_memory_swift_checkout()
    eval_binary = build_memory_eval_binary(repo_path)
    profile = load_or_create_profile()
    datasets = materialize_dataset_cache(upstream_evals_root(repo_path))
    embedding_baseline = seed_baseline_model("embedding", repo_path=repo_path)
    reranker_baseline, reranker_source = _seed_reranker_baseline(repo_path)

    summary = {
        "setup": SETUP_NAME,
        "memory_swift_repo": str(memory_swift_repo_path()),
        "memory_eval_binary": str(eval_binary),
        "hardware_profile": str(hardware_profile_path()),
        "datasets": {key: str(value) for key, value in datasets.items()},
        "baselines": {
            "embedding": str(embedding_baseline),
            "reranker": str(reranker_baseline),
        },
        "reranker_baseline_source": reranker_source,
    }
    print(json.dumps(summary, indent=2))
    print("---")
    print(f"setup:                  {SETUP_NAME}")
    print(f"memory_swift_repo:      {repo_path}")
    print(f"memory_eval_binary:     {eval_binary}")
    print(f"hardware_profile:       {hardware_profile_path()}")
    print(f"quick_eval_root:        {datasets['quick_eval']}")
    print(f"full_eval_root:         {datasets['full_eval']}")
    print(f"baseline_embedder:      {embedding_baseline}")
    print(f"baseline_reranker:      {reranker_baseline}")
    print(f"reranker_source:        {reranker_source}")
    print(f"reranker_batch_size:    {profile.reranker_batch_size}")
    print(f"datasets_root:          {datasets_root()}")


if __name__ == "__main__":
    main()
