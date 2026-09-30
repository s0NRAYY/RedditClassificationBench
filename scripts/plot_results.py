#!/usr/bin/env python3
"""Charts for the results page: all-community headline and names-vs-descriptions."""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

R = "results/"
KS = [4, 16, 64]
MODELS = [
    ("Jev 1.13", "jev-medium", "#2563eb"),
    ("Kev 0.8B", "kev-0.8b-medium", "#7c3aed"),
    ("Laya (base, zero-shot)", "laya-medium", "#ea580c"),
]
DESC, NAMES = "anonymous-id+description", "name-only"


def rows(folder, representation):
    return {m["k"]: m for m in json.load(open(R + folder + "/metrics.json"))
            if m["community_track"] == "all" and m["difficulty"] == "hard"
            and m["representation"] == representation}


def style(ax, title, subtitle):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_title(f"{title}\n", loc="left", fontsize=14, fontweight="bold")
    ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=10.5, color="#4b5563")


plt.rcParams.update({"font.family": "Helvetica Neue", "font.size": 11})

# 1. Headline over all communities. The supervised baseline only exists for seen communities
#    (unseen ones have no training posts), so it is a muted reference, labelled as such.
fig, ax = plt.subplots(figsize=(8.6, 5.2))
logreg = {m["k"]: m["accuracy"] for m in json.load(open(R + "baselines/logreg-seen-medium.json"))["metrics"]
          if m["difficulty"] == "hard"}
ax.plot(KS, [logreg[k] for k in KS], color="#9ca3af", linestyle="--", linewidth=1.8, marker="o", markersize=4)
for k in KS:
    ax.annotate(f"{logreg[k]:.0%}", (k, logreg[k]), textcoords="offset points", xytext=(0, 8), ha="center",
                color="#9ca3af", fontsize=9)
ax.annotate("Supervised logreg\n(seen communities only)", (KS[-1], logreg[KS[-1]]), textcoords="offset points",
            xytext=(14, 0), va="center", color="#9ca3af", fontsize=9.5, linespacing=1.2)
for name, folder, color in MODELS:
    r = rows(folder, DESC)
    acc = [r[k]["accuracy"] for k in KS]
    ax.fill_between(KS, [r[k]["accuracy_ci95_low"] for k in KS], [r[k]["accuracy_ci95_high"] for k in KS],
                    color=color, alpha=0.13, linewidth=0)
    ax.plot(KS, acc, color=color, linewidth=3, marker="o", markersize=7)
    for k, a in zip(KS, acc):
        ax.annotate(f"{a:.0%}", (k, a), textcoords="offset points", xytext=(0, 9), ha="center",
                    color=color, fontsize=10, fontweight="bold")
    ax.annotate(name, (KS[-1], acc[-1]), textcoords="offset points", xytext=(14, 0), va="center",
                color=color, fontsize=10.5, fontweight="bold")
ax.plot(KS, [1 / k for k in KS], color="#9ca3af", linestyle=":", linewidth=1.8)
ax.annotate("random guess", (16, 1 / 16), textcoords="offset points", xytext=(0, -15), ha="center",
            color="#6b7280", fontsize=9.5)
ax.set_xscale("log", base=2)
ax.set_xticks(KS, [str(k) for k in KS])
ax.set_xlim(3.3, 170)
ax.set_ylim(0, 0.82)
ax.set_xlabel("Number of candidate subreddits (log scale)")
ax.set_ylabel("Accuracy")
style(ax, "Headline: all 1,242 communities", "Post titles only · look-alike candidates · subreddit names hidden")
fig.tight_layout()
fig.savefig(R + "headline-all.png", dpi=200)

# 2. Names vs descriptions at hard K=16.
fig, ax = plt.subplots(figsize=(8.6, 4.8))
width = 0.36
for i, (name, folder, color) in enumerate(MODELS):
    for j, (rep, label, alpha) in enumerate([(NAMES, "names", 1.0), (DESC, "descriptions", 0.55)]):
        m = rows(folder, rep)[16]
        x = i + (j - 0.5) * width
        ax.bar(x, m["accuracy"], width * 0.92, color=color, alpha=alpha)
        ax.errorbar(x, m["accuracy"], yerr=[[m["accuracy"] - m["accuracy_ci95_low"]],
                                              [m["accuracy_ci95_high"] - m["accuracy"]]],
                    color="#374151", capsize=3, linewidth=1)
        ax.annotate(f"{m['accuracy']:.0%}\n{label}", (x, m["accuracy_ci95_high"]), textcoords="offset points",
                    xytext=(0, 4), ha="center", fontsize=9, color=color, fontweight="bold")
ax.axhline(1 / 16, color="#9ca3af", linestyle=":", linewidth=1.8)
ax.annotate("random guess", (0.5, 1 / 16), textcoords="offset points", xytext=(0, 4), ha="center",
            color="#6b7280", fontsize=9.5)
ax.set_xticks(range(len(MODELS)), [n for n, _, _ in MODELS])
ax.set_ylim(0, 0.62)
ax.set_ylabel("Accuracy (hard, 16 candidates)")
style(ax, "Does the model need subreddit names?",
      "Same posts and candidates, shown by name vs by description only")
fig.tight_layout()
fig.savefig(R + "names-vs-descriptions.png", dpi=200)
