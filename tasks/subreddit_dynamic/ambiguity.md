# Label ambiguity on the headline slice

How much of the headline slice (hard negatives, descriptions only) can be answered from a post title at all?

## Protocol

- Sample: 150 posts from the `medium` preset selection (seed `20260930`), 50 each at K = 4, 16, 64; hard negatives; `anonymous-id+description`; both community tracks (120 seen, 30 unseen). Candidates and display order are exactly what models see.
- Annotator: a strong general-purpose LLM, **not a human**. It sees only the title and the anonymous option descriptions, never the gold label, and returns (a) every option where a careful human could reasonably believe the post was published and (b) its single best option.
- Instruction: *"List EVERY option where a careful human could reasonably believe this post was published, given only the title. Include an option only if the title genuinely fits it; do not include options just because they are loosely related. Then give the single most likely option."*
- Per-item annotations (IDs and labels only, no Reddit text): `results/ambiguity/annotations.jsonl`. 95% intervals are bootstrap over items.

## Results

| K | Gold not defensible from the title | Plausible options per item | Gold is the only plausible option | Annotator's best pick = gold |
|---:|---:|---:|---:|---:|
| 4 | 0.02 [0.00, 0.06] | 2.0 | 0.28 | 0.80 [0.68, 0.90] |
| 16 | 0.04 [0.00, 0.10] | 4.2 | 0.08 | 0.68 [0.56, 0.80] |
| 64 | 0.10 [0.02, 0.18] | 5.8 | 0.04 | 0.64 [0.52, 0.78] |
| all | 0.05 [0.02, 0.09] | 4.0 | 0.13 | 0.71 [0.63, 0.78] |

## Reading

- About 5% of headline items (10% at K = 64) cannot be answered from the title: the gold community is not a defensible choice (e.g. a title like "The Legend of the Black Sun" posted to a Dark Souls trading subreddit). Accuracy above ~0.90 at K = 64 is therefore not attainable.
- Most items have several plausible communities. The gold is identifiable by *likelihood*, not by elimination, so the benchmark rewards calibrated judgement between overlapping communities rather than keyword matching.
- A strong LLM reading carefully scores 0.80 / 0.68 / 0.64. Treat that as a reference point for a strong reader, not a ceiling: decision models may exceed it.
- Limitations: one LLM annotator, no human agreement measurement, 50 items per K. A human spot-check of the "not defensible" items is the next step before quoting these numbers as a ceiling.
