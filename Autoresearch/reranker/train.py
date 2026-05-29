"""Mutable reranker experiment surface for Memory.swift autoresearch."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

AUTORESEARCH_ROOT = Path(__file__).resolve().parents[1]
if str(AUTORESEARCH_ROOT) not in sys.path:
    sys.path.insert(0, str(AUTORESEARCH_ROOT))

from memory_autoresearch.cache import (
    baseline_artifact_path,
    candidate_artifact_path,
    checkpoint_path,
    configure_setup,
    datasets_root,
    metrics_path,
    report_path,
)
from memory_autoresearch.checkpoints import (
    checkpoint_config,
    load_pretrained_weights,
    save_mlx_weights,
)
from memory_autoresearch.config import DEFAULT_TIME_BUDGET_SECONDS, MODEL_SPECS
from memory_autoresearch.data import load_retrieval_examples
from memory_autoresearch.eval import EvalSummary, evaluate_candidate
from memory_autoresearch.export import export_coreml_model
from memory_autoresearch.hardware import load_or_create_profile
from memory_autoresearch.modeling import RerankerModel
from memory_autoresearch.scoring import EvalMetrics, decide_candidate_status
from memory_autoresearch.tokenization import BertTokenizerAdapter
from memory_autoresearch.training import train_reranker
from memory_autoresearch.upstream import build_memory_eval_binary, prepare_memory_swift_checkout


# ---------------------------------------------------------------------------
# Mutable experiment controls
# ---------------------------------------------------------------------------

ACTIVE_COMPONENT = "reranker"
EVAL_PROFILE = "coreml_fast_rerank"
LEARNING_RATE = 2e-4
# Fast-rerank profile iteration: make the temporary profile's recall planner
# active in eval feature selection so its small semantic/lexical/rerank limits
# actually reach MemoryIndex. Use stable TinyBERT 1.6 calibration.
TIME_BUDGET_SECONDS = 0
RERANKER_CHECKPOINT = "cross-encoder/ms-marco-TinyBERT-L-2-v2"
RERANKER_LOGIT_SCALE = 1.6
SETUP_NAME = "reranker"
SETUP_ROOT = Path(__file__).resolve().parent
configure_setup(SETUP_NAME)


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
        "retrieval_train": root / "retrieval_train.jsonl",
        "quick_eval": root / "quick_eval",
        "full_eval": root / "full_eval",
    }
    missing = [name for name, path in paths.items() if not path.exists()]
    if missing:
        missing_str = ", ".join(missing)
        raise FileNotFoundError(
            f"Missing prepared assets: {missing_str}. Run `uv run reranker/prepare.py` first."
        )
    return paths


def _load_previous_metrics(component: str) -> EvalMetrics | None:
    report = _load_previous_report(component)
    if report is not None:
        aggregate = report.get("full", {}).get("aggregate")
        if isinstance(aggregate, dict):
            return EvalMetrics(**aggregate)
    path = metrics_path(component)
    if not path.exists():
        return None
    return EvalMetrics(**json.loads(path.read_text(encoding="utf-8")))


def _load_previous_report(component: str) -> dict | None:
    path = report_path(component)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _write_metrics(component: str, metrics: EvalMetrics) -> None:
    path = metrics_path(component)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(metrics), indent=2), encoding="utf-8")


def _clear_eval_provider_cache(dataset_root: Path) -> None:
    """Avoid stale reranker cache hits across different candidate artifacts."""
    provider_cache = dataset_root / "cache" / "provider"
    for path in provider_cache.glob("eval_provider_cache.sqlite*"):
        if path.is_file():
            path.unlink()


def _write_report(
    component: str, quick_summary: EvalSummary, full_summary: EvalSummary
) -> None:
    path = report_path(component)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "component": component,
        "profile": EVAL_PROFILE,
        "quick": quick_summary.asdict(),
        "full": full_summary.asdict(),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


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
    with (SETUP_ROOT / "results.tsv").open("a", encoding="utf-8") as handle:
        handle.write(line)
        handle.write("\n")


def _keep_candidate(
    component: str,
    artifact_path: Path,
    metrics: EvalMetrics,
    quick_summary: EvalSummary,
    full_summary: EvalSummary,
) -> None:
    baseline_path = baseline_artifact_path(component)
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    if baseline_path.exists():
        shutil.rmtree(baseline_path)
    shutil.copytree(artifact_path, baseline_path)
    _write_metrics(component, metrics)
    _write_report(component, quick_summary, full_summary)


def main() -> None:
    if ACTIVE_COMPONENT != "reranker":
        raise ValueError("The reranker setup only supports ACTIVE_COMPONENT='reranker'.")

    prepared = _ensure_prepared()
    hardware = load_or_create_profile()
    spec = MODEL_SPECS[ACTIVE_COMPONENT]
    checkpoint = RERANKER_CHECKPOINT
    tokenizer = BertTokenizerAdapter(
        checkpoint, max_sequence_length=spec.max_sequence_length
    )
    config = checkpoint_config(checkpoint, num_labels=8)
    config.vocab_size = tokenizer.vocab_size
    model = RerankerModel(config)
    load_pretrained_weights(model, ACTIVE_COMPONENT, checkpoint)
    model.classifier.weight = model.classifier.weight * RERANKER_LOGIT_SCALE
    if getattr(model.classifier, "bias", None) is not None:
        model.classifier.bias = model.classifier.bias * RERANKER_LOGIT_SCALE

    retrieval_examples = load_retrieval_examples(prepared["retrieval_train"])
    document_lookup = {
        example.positive_document_id: example.positive_document_text
        for example in retrieval_examples
    }
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
            "eval_profile": EVAL_PROFILE,
        },
    )

    artifact_path = export_coreml_model(
        ACTIVE_COMPONENT,
        result.model,
        config,
        candidate_artifact_path(ACTIVE_COMPONENT),
    )
    build_memory_eval_binary(prepare_memory_swift_checkout())
    _clear_eval_provider_cache(prepared["quick_eval"])
    _clear_eval_provider_cache(prepared["full_eval"])
    quick_summary, full_summary = evaluate_candidate(
        component=ACTIVE_COMPONENT,
        artifact_path=artifact_path,
        quick_dataset_root=prepared["quick_eval"],
        full_dataset_root=prepared["full_eval"],
        checkpoint=checkpoint,
        profile=EVAL_PROFILE,
    )
    quick_metrics = quick_summary.aggregate
    full_metrics = full_summary.aggregate

    baseline = _load_previous_metrics(ACTIVE_COMPONENT)
    baseline_report = _load_previous_report(ACTIVE_COMPONENT)
    status, decision_reason = decide_candidate_status(
        quick_metrics=quick_metrics,
        full_metrics=full_metrics,
        baseline_metrics=baseline,
        current_report={
            "component": ACTIVE_COMPONENT,
            "quick": quick_summary.asdict(),
            "full": full_summary.asdict(),
        },
        baseline_report=baseline_report,
    )
    keep = status == "keep"
    if keep:
        _keep_candidate(
            ACTIVE_COMPONENT, artifact_path, full_metrics, quick_summary, full_summary
        )
    _append_results_row(
        quick_metrics,
        status=status,
        description=(
            f"{ACTIVE_COMPONENT} profile={EVAL_PROFILE} ckpt={checkpoint} "
            f"lr={LEARNING_RATE} steps={result.steps} loss={result.average_loss:.4f} "
            f"reason={decision_reason}"
        ),
    )

    print("---")
    print(f"component:         {ACTIVE_COMPONENT}")
    print(f"profile:           {EVAL_PROFILE}")
    print(f"memory_score:      {quick_metrics.memory_score:.6f}")
    print(f"storage_score:     {quick_metrics.storage_score:.6f}")
    print(f"recall_score:      {quick_metrics.recall_score:.6f}")
    print(f"model_mb:          {quick_metrics.model_mb:.1f}")
    print(f"latency_ms:        {quick_metrics.latency_ms:.1f}")
    print(f"training_seconds:  {result.training_seconds:.1f}")
    print(f"num_steps:         {result.steps}")
    print(f"average_loss:      {result.average_loss:.6f}")
    print(f"status:            {status}")
    print(f"decision_reason:   {decision_reason}")
    for phase_name, summary in (("quick", quick_summary), ("full", full_summary)):
        print(f"{phase_name}_datasets:")
        for corpus_name, corpus_metrics in summary.corpora.items():
            print(
                "  "
                f"{corpus_name}: memory_score={corpus_metrics.memory_score:.6f} "
                f"storage_score={corpus_metrics.storage_score:.6f} "
                f"recall_score={corpus_metrics.recall_score:.6f} "
                f"latency_ms={corpus_metrics.latency_ms:.1f}"
            )
            if corpus_metrics.stage_latency_p95:
                stage_bits = " ".join(
                    f"{key}={value:.1f}"
                    for key, value in sorted(corpus_metrics.stage_latency_p95.items())
                )
                print(f"    stage_p95_ms: {stage_bits}")


if __name__ == "__main__":
    main()
