# Model Draft Review

Reviewed `scenarios.model_drafts.jsonl` for the agent-memory schema and replaced the original model output with six reviewable draft scenarios.

Fixes applied:

- Kept the file within the requested 5-8 scenario range.
- Ensured every row is a JSON object line with required agent-memory fields.
- Used only valid memory kinds, statuses, and facets from `PublicTypes.swift`.
- Removed brittle expectations around arbitrary handoff append behavior and temporal-only families.
- Kept recall queries only for write scenarios or no-write scenarios with setup memories.
- Marked the intentionally tricky hypothetical no-write case as `source_family` `pressure/model_review`.

Temporary eval status:

- Command: `swift run memory_eval run --profile nl_baseline --dataset-root /tmp/memory-agent-model-drafts-eval --no-cache --no-index-cache`
- Result: 6 draft scenarios, 100% recall Hit, but only 75% expected-write recall and 50% update-behavior accuracy.
- Failing drafts are useful pressure cases, not ready gate cases:
  - `model-commitment-resolution-caching` currently appends instead of merging status.
  - `model-pressure-hypothetical-profile-no-write` currently false-writes.
  - `model-recall-only-storage-decision` currently false-writes during recall-only interaction.

Keep this file as a review queue. Promote individual rows into `scenarios.generated.jsonl` or `scenarios.pressure.jsonl` only after their labels are confirmed and the intended runtime behavior is implemented.
