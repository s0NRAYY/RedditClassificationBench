# Social Routing Bench

The v1 benchmark contains one task: dynamic subreddit routing. A model receives a Reddit post and a runtime-defined candidate set, then selects the post's original community.

> **Public alpha:** contributors and benchmark feedback wanted. The worker protocol and supported adapters are usable, but this is not yet a finished universal benchmark.

The builder creates one paired post set. `title` and `title+selftext` are views over the same `post_id`; splits, targets, negative pools, and candidate permutations do not change between text modes.

## Build

The preferred raw archive contains `title`, a body column (`selftext`, `body`, or `text`), a subreddit column, and preferably a Reddit post ID or permalink. The upstream public archive currently contains only `label,text`; build that frozen title track explicitly with `--source-mode title-only`. Its `build.json` exposes only the `title` mode, so it cannot be mistaken for the paired track.

```bash
python -m pip install -e .
build-subreddit-dynamic \
  --posts path/to/raw/csvs \
  --communities path/to/req_subreddits.csv \
  --output data/subreddit_dynamic
```

For the published title archive:

```bash
build-subreddit-dynamic \
  --source-mode title-only \
  --posts data/raw/cleaned_all_title_data_top.csv \
  --communities data/raw/req_subreddits.csv \
  --output data/subreddit_dynamic
```

Outputs:

```text
posts.parquet
communities.parquet
manifests/negative_pools.parquet
manifests/instances.parquet
build.json
```

The embedding model and exact revision used for negative mining are frozen in `tasks/subreddit_dynamic/task.yaml`.

## Evaluate Decision Index models

The CLI resolves known [Jev Decision Index](https://huggingface.co/spaces/multimodalart/jev-decision-index) models to isolated runtime workers:

```bash
srb models list
srb doctor hf://AlexWortega/openjev
python -m pip install -e '.[openjev]'
srb run hf://AlexWortega/openjev \
  --dataset data/subreddit_dynamic \
  --output results/openjev
```

Built-in worker families cover System One HTTP, Surogate, OpenJev, Intern-Decision, Transformers models exposing `predict`, GLiNER2, and Laya. On the first HTTP run, the CLI securely prompts for the endpoint and API key, then saves them per model in `~/.config/social-routing-bench/credentials.json` with mode `0600`:

```bash
srb configure hf://jaredpalmer/kev-4b
srb doctor hf://jaredpalmer/kev-4b
srb run hf://jaredpalmer/kev-4b \
  --dataset data/subreddit_dynamic \
  --output results/kev-4b
```

`srb configure` changes saved credentials. The API key is masked while typing and reaches the worker only through its environment; it is never placed in process arguments or run metadata. An empty key supports unauthenticated local endpoints.

Jev runs through OpenRouter's native System One endpoint; only an OpenRouter API key is needed, the endpoint defaults to `https://openrouter.ai/api`:

```bash
srb configure typesafe/jev-1.13
srb run typesafe/jev-1.13 \
  --dataset data/subreddit_dynamic \
  --task-config tasks/subreddit_dynamic/medium.yaml \
  --output results/jev-medium
```

`~typesafe/jev-latest` is also registered but follows new releases; prefer the versioned ID for published results. HTTP runs record the version the server reports (for Jev, e.g. `typesafe/jev-1.13-20260917`) as `served_model` in `run.json`, and a resume against a different served version is refused. Transient 429/5xx responses are retried with backoff; OpenRouter's per-request `usage.cost` is recorded as `api_cost_usd`.

Any other model can use an external JSONL worker without adding its dependencies to the benchmark process:

```bash
srb run hf://owner/model \
  --command-worker "python my_worker.py" \
  --dataset data/subreddit_dynamic \
  --output results/model
```

The worker first answers `{"type":"hello","protocol":1}` with its limits, modalities, batching support, and probability source (`native`, `self_reported`, or `none`). Prediction messages contain `state` and the canonical typed `questions`; result messages return choices, optional probability distributions, usage, timing, and forward counts. See `tests/support_worker.py` for the complete minimal protocol.

Laya remains available through the same worker boundary:

```bash
python -m pip install -e '.[laya]'
evaluate-subreddit-dynamic \
  --dataset data/subreddit_dynamic \
  --config configs/laya.yaml \
  --output results/laya-multilingual
```

The evaluation is resumable at prediction granularity. It runs every available text mode across both candidate representations, seen/unseen communities, random/semantic/hard negatives, and every configured `K`.

Choose a matrix with `--task-config` (see `tasks/subreddit_dynamic/README.md` for sizes and timings): `smoke.yaml` checks plumbing, `medium.yaml` is the recommended comparison preset, and the default `task.yaml` is the full matrix. Posts per community come from `--posts-per-community`, then the preset, then the model config (ten by default). If a worker declares `max_options` below a requested `K`, that `K` is skipped with a warning and listed as `skipped_candidate_counts` in `run.json`; candidates are never truncated.

Before a full run, check the adapter with `srb sanity MODEL` (ten unambiguous hand-written posts; below 0.8 means the adapter or prompt format is suspect). The headline number is accuracy on hard negatives with descriptions only (names hidden), pooled over community tracks, at `K` = 16 and 64; see `tasks/subreddit_dynamic/README.md` and `tasks/subreddit_dynamic/ambiguity.md`.

Results contain `run.json`, `predictions.jsonl`, `metrics.json`, and `metrics.csv`. Accuracy, its 95% bootstrap interval, macro community accuracy, latency, token usage, forward count, and cost are always reported. NLL, multiclass Brier, ECE-15, and AURC are reported only for models that return probabilities; label-only models are never assigned fabricated confidence.

## Compare runs

Compatible runs can be compared by benchmark slice and exported as JSON:

```bash
srb compare results/laya-smoke results/gliner-smoke \
  --output results/smoke-comparison.json
```

The command rejects different dataset builds, task matrices, sampled posts, prediction IDs, or metric slices. Calibration fields remain unavailable for label-only models.

## Runtime notices

- Supported Python versions are 3.11–3.13. Python 3.14 is excluded because current Torch/GLiNER releases warn that their TorchScript path is unsupported.
- Set `HF_TOKEN` to remove the Hugging Face unauthenticated-download notice and receive higher rate limits.
- GLiNER2 may report that its custom `extractor` configuration is loading through Transformers and that DeBERTa falls back from SDPA to eager attention. Both messages come from the upstream runtime; the pinned GLiNER smoke inference is verified with this fallback.
- Runtime warnings are intentionally not suppressed globally. A new or changed warning should be treated as a compatibility signal.

## Reproducibility and provenance

- Source titles and community metadata come from [TheShadow29/subreddit-classification-dataset](https://github.com/TheShadow29/subreddit-classification-dataset), whose repository is MIT-licensed. Reddit content remains subject to Reddit's terms and the rights of its authors.
- The exact model and revision, adapter and worker protocol, probability provenance, runtime capabilities, environment, dataset build, candidate configuration, latency, and forward counts are recorded in run and prediction outputs.
- Model weights and caches are downloaded by their optional runtimes and are not redistributed here.

## License and citation

Benchmark code is released under the MIT License. See `LICENSE`. Citation metadata is available in `CITATION.cff`.
