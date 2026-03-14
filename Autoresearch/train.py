"""Mutable experiment surface for Memory.swift autoresearch."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import asdict
from pathlib import Path

from memory_autoresearch.cache import (
    baseline_artifact_path,
    candidate_artifact_path,
    checkpoint_path,
    datasets_root,
    metrics_path,
)
from memory_autoresearch.checkpoints import checkpoint_config, load_pretrained_weights, save_mlx_weights
from memory_autoresearch.config import DEFAULT_TIME_BUDGET_SECONDS, MODEL_SPECS
from memory_autoresearch.data import load_retrieval_examples, load_typing_examples
from memory_autoresearch.eval import evaluate_candidate
from memory_autoresearch.export import export_coreml_model
from memory_autoresearch.hardware import load_or_create_profile
from memory_autoresearch.modeling import EmbeddingModel, RerankerModel, TypingModel
from memory_autoresearch.scoring import EvalMetrics, should_keep_candidate
from memory_autoresearch.tokenization import BertTokenizerAdapter
from memory_autoresearch.training import train_embedding, train_reranker, train_typing


# ---------------------------------------------------------------------------
# Mutable experiment controls
# ---------------------------------------------------------------------------

ACTIVE_COMPONENT = "typing"
LEARNING_RATE = 3e-4
TIME_BUDGET_SECONDS = DEFAULT_TIME_BUDGET_SECONDS
TYPING_CHECKPOINT_OVERRIDE = "google/bert_uncased_L-2_H-128_A-2"


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _ensure_prepared() -> dict[str, Path]:
    root = datasets_root()
    paths = {
        "typing_train": root / "typing_train.jsonl",
        "retrieval_train": root / "retrieval_train.jsonl",
        "quick_eval": root / "quick_eval",
        "full_eval": root / "full_eval",
    }
    missing = [name for name, path in paths.items() if not path.exists()]
    if missing:
        missing_str = ", ".join(missing)
        raise FileNotFoundError(
            f"Missing prepared assets: {missing_str}. Run `uv run prepare.py` first."
        )
    return paths


def _load_previous_metrics(component: str) -> EvalMetrics | None:
    path = metrics_path(component)
    if not path.exists():
        return None
    return EvalMetrics(**json.loads(path.read_text(encoding="utf-8")))


def _write_metrics(component: str, metrics: EvalMetrics) -> None:
    path = metrics_path(component)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(metrics), indent=2), encoding="utf-8")


def _append_results_row(metrics: EvalMetrics, status: str, description: str) -> None:
    line = "\t".join(
        [
            _git_commit(),
            metrics.component,
            f"{metrics.memory_score:.6f}",
            f"{metrics.storage_score:.6f}",
            f"{metrics.recall_score:.6f}",
            f"{metrics.model_mb:.1f}",
            f"{metrics.latency_ms:.1f}",
            status,
            description,
        ]
    )
    with open("results.tsv", "a", encoding="utf-8") as handle:
        handle.write(line)
        handle.write("\n")


def _keep_candidate(component: str, artifact_path: Path, metrics: EvalMetrics) -> None:
    baseline_path = baseline_artifact_path(component)
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    if baseline_path.exists():
        shutil.rmtree(baseline_path)
    shutil.copytree(artifact_path, baseline_path)
    _write_metrics(component, metrics)


def _build_model(component: str, vocab_size: int):
    checkpoint = _resolved_checkpoint(component)
    config = checkpoint_config(checkpoint, num_labels=8)
    config.vocab_size = vocab_size
    if component == "typing":
        return TypingModel(config), config
    if component == "embedding":
        return EmbeddingModel(config), config
    if component == "reranker":
        return RerankerModel(config), config
    raise ValueError(f"Unsupported component: {component}")


def _resolved_checkpoint(component: str) -> str:
    if component == "typing" and TYPING_CHECKPOINT_OVERRIDE:
        return TYPING_CHECKPOINT_OVERRIDE
    return MODEL_SPECS[component].checkpoint


def main() -> None:
    if ACTIVE_COMPONENT not in MODEL_SPECS:
        raise ValueError(f"ACTIVE_COMPONENT must be one of {sorted(MODEL_SPECS)}")

    prepared = _ensure_prepared()
    hardware = load_or_create_profile()
    spec = MODEL_SPECS[ACTIVE_COMPONENT]
    checkpoint = _resolved_checkpoint(ACTIVE_COMPONENT)
    tokenizer = BertTokenizerAdapter(checkpoint, max_sequence_length=spec.max_sequence_length)
    model, config = _build_model(ACTIVE_COMPONENT, tokenizer.vocab_size)
    load_pretrained_weights(model, ACTIVE_COMPONENT, checkpoint)

    if ACTIVE_COMPONENT == "typing":
        examples = load_typing_examples(prepared["typing_train"])
        result = train_typing(
            model=model,
            tokenizer=tokenizer,
            examples=examples,
            hardware=hardware,
            time_budget_seconds=TIME_BUDGET_SECONDS,
            learning_rate=LEARNING_RATE,
        )
    else:
        retrieval_examples = load_retrieval_examples(prepared["retrieval_train"])
        document_lookup = {
            example.positive_document_id: example.positive_document_text
            for example in retrieval_examples
        }
        if ACTIVE_COMPONENT == "embedding":
            result = train_embedding(
                model=model,
                tokenizer=tokenizer,
                examples=retrieval_examples,
                document_lookup=document_lookup,
                hardware=hardware,
                time_budget_seconds=TIME_BUDGET_SECONDS,
                learning_rate=LEARNING_RATE,
            )
        else:
            result = train_reranker(
                model=model,
                tokenizer=tokenizer,
                examples=retrieval_examples,
                document_lookup=document_lookup,
                hardware=hardware,
                time_budget_seconds=TIME_BUDGET_SECONDS,
                learning_rate=LEARNING_RATE,
            )

    weight_path = checkpoint_path(ACTIVE_COMPONENT)
    save_mlx_weights(
        result.model,
        weight_path,
        metadata={
            "component": ACTIVE_COMPONENT,
            "checkpoint": checkpoint,
            "training_seconds": f"{result.training_seconds:.2f}",
            "steps": str(result.steps),
            "average_loss": f"{result.average_loss:.6f}",
        },
    )

    artifact_path = export_coreml_model(ACTIVE_COMPONENT, result.model, config, candidate_artifact_path(ACTIVE_COMPONENT))
    quick_metrics, full_metrics = evaluate_candidate(
        component=ACTIVE_COMPONENT,
        artifact_path=artifact_path,
        quick_dataset_root=prepared["quick_eval"],
        full_dataset_root=prepared["full_eval"],
        checkpoint=checkpoint,
    )

    baseline = _load_previous_metrics(ACTIVE_COMPONENT)
    quick_keep = should_keep_candidate(quick_metrics, baseline)
    full_keep = should_keep_candidate(full_metrics, baseline)
    keep = quick_keep and full_keep
    if keep:
        status = "keep"
    elif quick_keep and not full_keep:
        status = "discard_full"
    else:
        status = "discard"
    if keep:
        _keep_candidate(ACTIVE_COMPONENT, artifact_path, full_metrics)
    _append_results_row(
        quick_metrics,
        status=status,
        description=(
            f"{ACTIVE_COMPONENT} ckpt={checkpoint} "
            f"lr={LEARNING_RATE} steps={result.steps} loss={result.average_loss:.4f}"
        ),
    )

    print("---")
    print(f"component:         {ACTIVE_COMPONENT}")
    print(f"memory_score:      {quick_metrics.memory_score:.6f}")
    print(f"storage_score:     {quick_metrics.storage_score:.6f}")
    print(f"recall_score:      {quick_metrics.recall_score:.6f}")
    print(f"model_mb:          {quick_metrics.model_mb:.1f}")
    print(f"latency_ms:        {quick_metrics.latency_ms:.1f}")
    print(f"training_seconds:  {result.training_seconds:.1f}")
    print(f"num_steps:         {result.steps}")
    print(f"average_loss:      {result.average_loss:.6f}")
    print(f"status:            {status}")


if __name__ == "__main__":
    main()
