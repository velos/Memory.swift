#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  run_eval.sh profile <profile> [--index-cache]
  run_eval.sh all [--index-cache]
  run_eval.sh compare <run-a.json> <run-b.json> [more runs...]
  run_eval.sh gate <baseline.json> <run.json> [more runs...]
  run_eval.sh diagnose-longmemeval <source-run.json> [--scope misses|misses-and-partials|all] [--wide-limit N] [--no-index-cache]

Environment:
  DATASET_ROOT   Dataset root path (default: ./Evals/general_v2 for run modes, ./Evals/longmemeval_v2 for diagnose-longmemeval)
  PROFILE        Eval profile for diagnose-longmemeval (default: coreml_default)

Notes:
  - Default behavior is deterministic: --no-cache --no-index-cache.
  - Pass --index-cache to reuse cached indexes.
EOF
}

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

MODE="$1"
shift

DATASET_ROOT="${DATASET_ROOT:-}"
PROFILE="${PROFILE:-coreml_default}"
NO_PROVIDER_CACHE_FLAG="--no-cache"
INDEX_CACHE_FLAG="--no-index-cache"

if [[ "${1:-}" == "--index-cache" ]]; then
  INDEX_CACHE_FLAG="--index-cache"
  shift
fi

case "$MODE" in
  profile)
    if [[ $# -lt 1 ]]; then
      usage
      exit 1
    fi
    PROFILE="$1"
    shift
    if [[ "${1:-}" == "--index-cache" ]]; then
      INDEX_CACHE_FLAG="--index-cache"
      shift
    elif [[ "${1:-}" == "--no-index-cache" ]]; then
      INDEX_CACHE_FLAG="--no-index-cache"
      shift
    fi
    EFFECTIVE_DATASET_ROOT="${DATASET_ROOT:-./Evals/general_v2}"
    swift run memory_eval run \
      --profile "$PROFILE" \
      --dataset-root "$EFFECTIVE_DATASET_ROOT" \
      "$NO_PROVIDER_CACHE_FLAG" \
      "$INDEX_CACHE_FLAG" \
      "$@"
    ;;
  all)
    EFFECTIVE_DATASET_ROOT="${DATASET_ROOT:-./Evals/general_v2}"
    swift run memory_eval run \
      --dataset-root "$EFFECTIVE_DATASET_ROOT" \
      "$NO_PROVIDER_CACHE_FLAG" \
      "$INDEX_CACHE_FLAG" \
      "$@"
    ;;
  compare)
    if [[ $# -lt 2 ]]; then
      usage
      exit 1
    fi
    swift run memory_eval compare "$@"
    ;;
  gate)
    if [[ $# -lt 2 ]]; then
      usage
      exit 1
    fi
    BASELINE="$1"
    shift
    swift run memory_eval gate --baseline "$BASELINE" "$@"
    ;;
  diagnose-longmemeval)
    if [[ $# -lt 1 ]]; then
      usage
      exit 1
    fi
    SOURCE_RUN="$1"
    shift
    DIAG_INDEX_CACHE_FLAG="--index-cache"
    if [[ "${1:-}" == "--no-index-cache" ]]; then
      DIAG_INDEX_CACHE_FLAG="--no-index-cache"
      shift
    fi
    EFFECTIVE_DATASET_ROOT="${DATASET_ROOT:-./Evals/longmemeval_v2}"
    swift run memory_eval diagnose-longmemeval \
      --profile "$PROFILE" \
      --dataset-root "$EFFECTIVE_DATASET_ROOT" \
      --source-run "$SOURCE_RUN" \
      "$DIAG_INDEX_CACHE_FLAG" \
      "$@"
    ;;
  *)
    usage
    exit 1
    ;;
esac
