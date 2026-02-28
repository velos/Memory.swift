#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  run_eval.sh profile <profile> [--index-cache]
  run_eval.sh all [--index-cache]
  run_eval.sh compare <run-a.json> <run-b.json> [more runs...]

Environment:
  DATASET_ROOT   Dataset root path (default: ./Evals)

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

DATASET_ROOT="${DATASET_ROOT:-./Evals}"
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
    swift run memory_eval run \
      --profile "$PROFILE" \
      --dataset-root "$DATASET_ROOT" \
      "$NO_PROVIDER_CACHE_FLAG" \
      "$INDEX_CACHE_FLAG" \
      "$@"
    ;;
  all)
    swift run memory_eval run \
      --dataset-root "$DATASET_ROOT" \
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
  *)
    usage
    exit 1
    ;;
esac
