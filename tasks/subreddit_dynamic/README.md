# Dynamic Subreddit Routing

## Input

Each evaluation instance combines one canonical post with one candidate pool. Text is rendered as either `title` or `title + "\n\n" + selftext`. Candidate representations are `name-only` or `anonymous-id+description`.

Only posts with a non-empty title and usable selftext enter the paired set. Empty, `[deleted]`, `[removed]`, and URL-only bodies are rejected. Exact normalized-title duplicates are removed globally before splitting, which also removes exact-title crossposts. V1 does not heuristically strip AutoModerator boilerplate or fuzzy-match near duplicates: the raw archive lacks reliable authorship metadata, and an unvalidated text heuristic would silently alter legitimate posts.

The public upstream archive has only `label,text`. A build made with `--source-mode title-only` is a separately marked title track: `selftext` stays empty and `title+selftext` evaluation is rejected. This preserves provenance instead of pretending the cleaned title is full text.

## Splits

Communities are deterministically assigned to `seen` or `unseen` before post splitting and negative mining. Seen communities have post-level train/dev/test splits. Every unseen-community post is test-only.

## Negative pools

Each `(community_track, target, difficulty)` has one frozen pool:

- `random`: deterministic permutation of every other community in the track.
- `semantic`: deterministic permutation of the nearest configured semantic window.
- `hard`: exact cosine-similarity ranking.

Seen vectors are means of embeddings from seen-train `title+selftext` only. Unseen vectors use public descriptions only. Stored mining strategies therefore distinguish `hard_content`, `semantic_content`, `hard_description`, and `semantic_description`.

For candidate count `K`, take the target and the first `K-1` negatives. Their display permutation is derived from `(benchmark_seed, post_id, difficulty, K)`; candidate sets are nested even though display order may change.

## Presets

All presets share `benchmark_seed: 20260926`, so the same posts and candidate sets are selected for every model. Counts below are for the title-only build (1,242 communities, one text mode, two representations).

| Preset | Posts / community | Difficulties | `K` | Posts | Predictions | Laya, M3 Max |
|---|---|---|---|---:|---:|---:|
| `smoke.yaml` | 1 (via CLI) | random | 2 | 1,242 | 2,484 | 22 s |
| `medium.yaml` | 2 | random, semantic, hard | 4, 16, 64 | 2,484 | 44,712 | 8 m 8 s |
| `task.yaml` | 10 (model config) | random, semantic, hard | 4, 8, 16, 32, 64 | 12,384 | 371,520 | not re-timed |

`medium` is the comparison preset: it keeps every track, difficulty, and representation and the smallest, middle, and largest `K`. Laya timing: Python 3.13.15, `laya-mlx` 0.2.0, MLX float16; slower runtimes scale roughly with predictions and `K`.

## Headline metric and tracks

Metrics are reported per `community_track` (`seen`, `unseen`) and pooled as `all`. For zero-shot models neither track was seen in training, so the split mainly reflects how hard negatives were mined (seen-train content vs public descriptions), not generalization; use `all` unless a model was trained on the seen-train split.

The headline number is accuracy on `all` tracks, `hard` negatives, `anonymous-id+description`, at K = 16 and 64. Hiding subreddit names separates understanding what a community is about from recognising its name, and hard negatives are where models differ. `srb run` prints it at the end of a run and `srb compare` puts it first. See `ambiguity.md` for how much of this slice is answerable from a title.

## Adapter sanity check

`srb sanity MODEL` sends ten hand-written, unambiguous posts (`sanity.yaml`) through the benchmark's exact question format in both representations. A model below 0.8 accuracy there most likely has an adapter or prompt-format problem; do not publish its benchmark results until that is resolved.

## Stored contracts

`posts.parquet`: `post_id`, `subreddit`, `title`, `selftext`, `created_utc`, `split`.

`communities.parquet`: `community_id`, `name`, `public_description`, `split_group`.

`negative_pools.parquet`: `pool_id`, `community_track`, `target`, `difficulty`, `mining_strategy`, `negative_pool`.

`instances.parquet`: `post_id`, `target`, `community_track`, `split`, `difficulty`, `pool_id`, `shuffle_seed`.
