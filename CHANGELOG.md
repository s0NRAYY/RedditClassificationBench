# Changelog

## Unreleased

- `medium` task preset (`tasks/subreddit_dynamic/medium.yaml`): 2 posts per community, all difficulties, `K` ∈ {4, 16, 64}.
- Task presets may set `posts_per_community` (CLI overrides preset, preset overrides model config).
- Candidate counts above a worker's declared `max_options` are skipped and recorded as `skipped_candidate_counts` in `run.json` instead of failing mid-run.

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
