# Social Routing Bench

The v1 benchmark contains one task: dynamic subreddit routing. A model receives a Reddit post and a runtime-defined candidate set, then selects the post's original community.

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

## Evaluate Laya

```bash
python -m pip install -e '.[laya]'
evaluate-subreddit-dynamic \
  --dataset data/subreddit_dynamic \
  --config configs/laya.yaml \
  --output results/laya-multilingual
```

The evaluation is resumable at prediction granularity. It runs every available text mode across both candidate representations, seen/unseen communities, random/semantic/hard negatives, and every configured `K`. The fixed sample contains up to ten test posts per community.

Results contain:

```text
run.json
predictions.jsonl
metrics.json
metrics.csv
```

Metrics include accuracy with a 95% bootstrap interval, macro community accuracy, NLL, multiclass Brier score, ECE-15, AURC, amortized P50/P95 latency, input tokens, and local API cost. Laya's public probabilities are rounded to four decimals; the evaluator records their original sum and renormalizes them before calibration metrics.

## Reproducibility and provenance

- Source titles and community metadata come from [TheShadow29/subreddit-classification-dataset](https://github.com/TheShadow29/subreddit-classification-dataset), whose repository is MIT-licensed. Reddit content remains subject to Reddit's terms and the rights of its authors.
- Laya evaluation uses [`laya-mlx`](https://github.com/mizorewww/laya-mlx) and `aac6fef/laya-multilingual-mlx`, both distributed under Apache-2.0. Model weights and caches are downloaded locally and are not redistributed here.
- Dataset rows, model revision, embedding revision, sampling seed, runtime version, environment, and candidate configuration are recorded in the generated build and run metadata.

## License and citation

Benchmark code is released under the MIT License. See `LICENSE`. Citation metadata is available in `CITATION.cff`.
