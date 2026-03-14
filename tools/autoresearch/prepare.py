"""Fixed bootstrapper for the Memory.swift autoresearch MLX loop."""

from __future__ import annotations

import json
from pathlib import Path

from memory_autoresearch.cache import (
    baseline_artifact_path,
    datasets_root,
    hardware_profile_path,
    memory_swift_repo_path,
)
from memory_autoresearch.data import materialize_dataset_cache
from memory_autoresearch.hardware import load_or_create_profile
from memory_autoresearch.upstream import (
    build_memory_eval_binary,
    prepare_memory_swift_checkout,
    seed_baseline_models,
    upstream_evals_root,
)


def main() -> None:
    repo_path = prepare_memory_swift_checkout()
    eval_binary = build_memory_eval_binary(repo_path)
    profile = load_or_create_profile()
    datasets = materialize_dataset_cache(upstream_evals_root(repo_path))
    baselines = seed_baseline_models(repo_path)

    summary = {
        "memory_swift_repo": str(memory_swift_repo_path()),
        "memory_eval_binary": str(eval_binary),
        "hardware_profile": str(hardware_profile_path()),
        "datasets": {key: str(value) for key, value in datasets.items()},
        "baselines": {key: str(value) for key, value in baselines.items()},
        "typing_baseline_expected_at": str(baseline_artifact_path("typing")),
    }
    print(json.dumps(summary, indent=2))
    print("---")
    print(f"memory_swift_repo:      {repo_path}")
    print(f"memory_eval_binary:     {eval_binary}")
    print(f"hardware_profile:       {hardware_profile_path()}")
    print(f"quick_eval_root:        {datasets['quick_eval']}")
    print(f"full_eval_root:         {datasets['full_eval']}")
    print(f"baseline_embedder:      {baselines['embedding']}")
    print(f"baseline_reranker:      {baselines['reranker']}")
    print(f"typing_batch_size:      {profile.typing_batch_size}")
    print(f"embedding_batch_size:   {profile.embedding_batch_size}")
    print(f"reranker_batch_size:    {profile.reranker_batch_size}")


if __name__ == "__main__":
    main()
