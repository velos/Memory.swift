"""Fixed support package for the Memory.swift autoresearch experiment loop."""

from .config import (
    ACTIVE_COMPONENTS,
    AUTORESEARCH_ROOT,
    COMPONENT_GATES,
    DEFAULT_MAX_SEQUENCE_LENGTH,
    DEFAULT_TIME_BUDGET_SECONDS,
    MEMORY_SWIFT_REPO_ROOT,
    MODEL_SPECS,
    STORAGE_SCORE_WEIGHTS,
    RETRIEVAL_SCORE_WEIGHTS,
)
from .scoring import (
    EvalMetrics,
    build_memory_score,
    compute_recall_score,
    compute_storage_score,
    should_keep_candidate,
)

__all__ = [
    "ACTIVE_COMPONENTS",
    "AUTORESEARCH_ROOT",
    "COMPONENT_GATES",
    "DEFAULT_MAX_SEQUENCE_LENGTH",
    "DEFAULT_TIME_BUDGET_SECONDS",
    "EvalMetrics",
    "MEMORY_SWIFT_REPO_ROOT",
    "MODEL_SPECS",
    "RETRIEVAL_SCORE_WEIGHTS",
    "STORAGE_SCORE_WEIGHTS",
    "build_memory_score",
    "compute_recall_score",
    "compute_storage_score",
    "should_keep_candidate",
]
