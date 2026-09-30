import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "/Users/khan/Documents/GitHub/RedditClassificationBench/results/"
KS = [4, 16, 64]
# Seen communities only, so the supervised baseline (which needs training posts) is comparable.
SLICE = {"community_track": "seen", "difficulty": "hard", "representation": "anonymous-id+description"}


def model_rows(folder):
    return {m["k"]: m for m in json.load(open(R + folder + "/metrics.json"))
            if all(m[f] == v for f, v in SLICE.items())}


logreg = json.load(open(R + "baselines/logreg-seen-medium.json"))
LINES = [
    ("Supervised baseline\n(logreg on title embeddings)",
     {m["k"]: m for m in logreg["metrics"] if m["difficulty"] == "hard"}, "#16a34a", "--", 0),
    ("Jev 1.13\n(build 2026-09-17)", model_rows("jev-medium"), "#2563eb", "-", 0),
    ("Kev 0.8B\n(zero-shot, local MLX)", model_rows("kev-0.8b-medium"), "#7c3aed", "-", 0),
]

plt.rcParams.update({"font.family": "Helvetica Neue", "font.size": 11})
fig, ax = plt.subplots(figsize=(8.6, 5.4))

for name, rows, color, style, label_dy in LINES:
    acc = [rows[k]["accuracy"] for k in KS]
    ax.fill_between(KS, [rows[k]["accuracy_ci95_low"] for k in KS],
                    [rows[k]["accuracy_ci95_high"] for k in KS], color=color, alpha=0.13, linewidth=0)
    ax.plot(KS, acc, color=color, linewidth=3 if style == "-" else 2.2, linestyle=style,
            marker="o", markersize=7 if style == "-" else 5)
    for k, a in zip(KS, acc):
        ax.annotate(f"{a:.0%}", (k, a), textcoords="offset points", xytext=(0, 9),
                    ha="center", color=color, fontsize=10, fontweight="bold")
    ax.annotate(name, (KS[-1], acc[-1]), textcoords="offset points", xytext=(14, label_dy),
                color=color, fontsize=10.5, fontweight="bold", va="center", linespacing=1.2)

ax.plot(KS, [1 / k for k in KS], color="#9ca3af", linestyle=":", linewidth=1.8)
ax.annotate("random guess", (16, 1 / 16), textcoords="offset points", xytext=(0, -15),
            ha="center", color="#6b7280", fontsize=9.5)

ax.set_xscale("log", base=2)
ax.set_xticks(KS, [str(k) for k in KS])
ax.set_xlim(3.3, 170)
ax.set_ylim(0, 0.85)
ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
ax.set_xlabel("Number of candidate subreddits (log scale)")
ax.set_ylabel("Accuracy")
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
ax.grid(axis="y", alpha=0.25)

fig.text(0.07, 0.95, "Which subreddit was this post from?", fontsize=15, fontweight="bold")
fig.text(0.07, 0.905, "Post titles only · look-alike candidates · subreddit names hidden",
         fontsize=11, color="#4b5563")
fig.text(0.07, 0.025,
         "Social Routing Bench · 1,988 titles from 994 communities with training data · "
         "shaded = 95% CI · run 2026-09-29/30",
         fontsize=8.5, color="#6b7280")
fig.tight_layout(rect=(0, 0.045, 1, 0.88))
fig.savefig(R + "headline.png", dpi=200)
for name, rows, *_ in LINES:
    print(name.split("\n")[0], [round(rows[k]["accuracy"], 3) for k in KS], rows[4].get("examples"))
