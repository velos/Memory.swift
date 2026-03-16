from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

DEFAULT_TIME_BUDGET_SECONDS = 300
DEFAULT_MAX_SEQUENCE_LENGTH = 512
RANDOM_SEED = 42
CACHE_NAMESPACE = "memory-swift-autoresearch"

ACTIVE_COMPONENTS = ("typing", "embedding", "reranker")
SUPPORTED_CORPORA = ("general", "tech", "scifact", "nfcorpus", "longmemeval", "typing_gold_v1")
TRAINING_CORPORA = ("general", "tech", "scifact", "nfcorpus")
PRIMARY_EVAL_CORPORA = ("general", "longmemeval")
SECONDARY_EVAL_CORPORA = ("tech",)
STRESS_EVAL_CORPORA = ("scifact", "nfcorpus")
TYPING_GOLD_CORPORA = ("typing_gold_v1",)
QUICK_EVAL_CORPORA = ("general", "longmemeval", "typing_gold_v1")
FULL_EVAL_CORPORA = ("general", "longmemeval", "tech", "typing_gold_v1")

MEMORY_TYPES = (
    "factual",
    "procedural",
    "episodic",
    "semantic",
    "emotional",
    "social",
    "contextual",
    "temporal",
)
MEMORY_TYPE_TO_INDEX = {name: index for index, name in enumerate(MEMORY_TYPES)}
INDEX_TO_MEMORY_TYPE = {index: name for name, index in MEMORY_TYPE_TO_INDEX.items()}


@dataclass(frozen=True)
class ModelSpec:
    component: str
    checkpoint: str
    artifact_name: str
    output_name: str
    benchmark_label: str
    hidden_size: int = 384
    intermediate_size: int = 1536
    num_layers: int = 6
    num_heads: int = 6
    max_sequence_length: int = DEFAULT_MAX_SEQUENCE_LENGTH


MODEL_SPECS = {
    "typing": ModelSpec(
        component="typing",
        checkpoint="nreimers/MiniLM-L6-H384-uncased",
        artifact_name="memory-type-classifier.mlpackage",
        output_name="type_logits",
        benchmark_label="typing",
    ),
    "embedding": ModelSpec(
        component="embedding",
        checkpoint="sentence-transformers/all-MiniLM-L6-v2",
        artifact_name="memory-embedder.mlpackage",
        output_name="embedding",
        benchmark_label="embedding",
    ),
    "reranker": ModelSpec(
        component="reranker",
        checkpoint="cross-encoder/ms-marco-MiniLM-L-6-v2",
        artifact_name="memory-reranker.mlpackage",
        output_name="relevance_score",
        benchmark_label="reranker",
    ),
}

STORAGE_SCORE_WEIGHTS = {
    "type_accuracy": 0.50,
    "macro_f1": 0.30,
    "span_coverage": 0.20,
}

RETRIEVAL_SCORE_WEIGHTS = {
    "mrr": 0.45,
    "ndcg": 0.30,
    "recall": 0.15,
    "hit_rate": 0.10,
}

MEMORY_SCORE_WEIGHTS = {
    "typing": {"storage": 0.70, "recall": 0.30},
    "embedding": {"storage": 0.20, "recall": 0.80},
    "reranker": {"storage": 0.20, "recall": 0.80},
}

COMPONENT_GATES = {
    "typing": {"model_mb": 10.0, "latency_ms": 25.0},
    "embedding": {"model_mb": 35.0, "latency_ms": 50.0},
    "reranker": {"model_mb": 25.0, "latency_ms": 150.0},
}

EXPORT_QUANTIZATION = {
    "typing": None,
    "embedding": "int4",
    "reranker": "int4",
}

HYSTERESIS_MIN_IMPROVEMENT = 0.003
HYSTERESIS_TIE_TOLERANCE = 0.001
PRIMARY_LATENCY_TOLERANCE = {
    "general": 0.15,
    "longmemeval": 0.20,
}
MODEL_SIZE_TOLERANCE_MULTIPLIER = 1.25
COMPONENT_CORPUS_SCORE_WEIGHTS = {
    "typing": {
        "general": 0.25,
        "longmemeval": 0.20,
        "tech": 0.10,
        "typing_gold_v1": 0.45,
    },
    "embedding": {
        "general": 0.45,
        "longmemeval": 0.45,
        "tech": 0.10,
    },
    "reranker": {
        "general": 0.45,
        "longmemeval": 0.45,
        "tech": 0.10,
    },
}

DEFAULT_QUICK_PROFILE_NAME = "automemory_quick"
DEFAULT_FULL_PROFILE_NAME = "automemory_full"
DEFAULT_MEMORY_EVAL_PROFILE = "coreml_rerank"
DEFAULT_RERANK_K_VALUES = "1,3,5,10"

AUTORESEARCH_ROOT = Path(__file__).resolve().parent.parent
MEMORY_SWIFT_REPO_ROOT = AUTORESEARCH_ROOT.parent
REPO_ROOT = AUTORESEARCH_ROOT
