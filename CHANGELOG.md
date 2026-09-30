# Changelog

## Unreleased

- `medium` task preset (`tasks/subreddit_dynamic/medium.yaml`): 2 posts per community, all difficulties, `K` ∈ {4, 16, 64}.
- Task presets may set `posts_per_community` (CLI overrides preset, preset overrides model config).
- Candidate counts above a worker's declared `max_options` are skipped and recorded as `skipped_candidate_counts` in `run.json` instead of failing mid-run.
- Jev (`typesafe/jev-1.13`, `~typesafe/jev-latest`) via OpenRouter's System One endpoint; hosted-API registry entries may omit a weights revision.
- HTTP runs record the server-reported model version as `served_model` in `run.json`, retry transient 429/5xx with backoff, and record provider-reported cost as `api_cost_usd`.
- `srb sanity MODEL`: hand-written, unambiguous examples in the exact benchmark question format to catch adapter bugs before a run. It caught the GLiNER2 adapter folding label descriptions into label names (description-mode accuracy on obvious examples 0.5 → 1.0 after using GLiNER2's native label descriptions); earlier GLiNER description results are invalid.
- Metrics add a pooled `community_track: all`; the headline metric (hard, descriptions only, all tracks, K = 16/64) is printed by `srb run` and first in `srb compare`.
- `srb compare` accepts runs whose worker limits skipped some `K`, comparing the shared counts.
- Label-ambiguity estimate for the headline slice (`tasks/subreddit_dynamic/ambiguity.md`, LLM-annotated).
- `predictions.jsonl` rows now keep the full record: ordered candidates, chosen label, the complete probability vector, any extra model output (e.g. confidence) and the raw per-request response metadata and usage. New metrics (ranks, top-k, recalibration) can be computed from existing runs without rerunning.
- Supervised reference baseline (`scripts/logreg_baseline.py`): logistic regression on `BAAI/bge-small-en-v1.5` title embeddings trained on seen-train posts; seen track only, since unseen communities have no training posts. `medium` hard: 0.718 / 0.593 / 0.537 at K = 4 / 16 / 64.

## v0.3.0 — Public alpha

Positioning: **public alpha — contributors and benchmark feedback wanted**. This is not yet a finished universal benchmark.

### Added

- Isolated JSONL worker protocol with declared capabilities and probability provenance.
- Registry-backed adapters for Laya, OpenJev, GLiNER2, Intern-Decision, Bosun-style Transformers models, System One HTTP, and Surogate HTTP.
- Rich terminal UI with model tables, runtime checks, benchmark configuration, progress, and resumable runs.
- Per-model HTTP endpoint and API-key storage with masked prompts and `0600` permissions.
- `srb compare` for compatible run directories and JSON comparison exports.
- Smoke task preset at `tasks/subreddit_dynamic/smoke.yaml`.

### Changed

- Every registered Hugging Face model is pinned to an immutable commit SHA.
- Run metadata records worker revision status; HTTP servers explicitly report unresolved server revisions when unavailable.
- Calibration metrics are omitted for models without probabilities instead of fabricating confidence.
- Supported Python versions are 3.11–3.13.

### Known limitations

- Registered adapters cover multiple Decision Index runtime families, not every listed model.
- HTTP model identity depends on what the remote server reports.
- GLiNER2 emits documented upstream Transformers and eager-attention fallback notices.
