"""Topic-baseline figure (05_topic_baseline.py) -> figures/topic_baseline.png"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import (EDM, TRAIN_CSV, META, IMAGES, DISTRACTORS, MISC_ZIP, RESULTS, FIGURES, OUTPUTS, ASSETS,
                    SEED as CFG_SEED, GROUP_ENCODER, GROUP_K, emb)
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

R = pd.read_csv(RESULTS / "topic_baseline.csv")
FIG = FIGURES
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False})
GROUPS = [("8 groups", "Eedi level-2 topic, 7 largest + other", "Semantic MiniLM (k=8)", "Semantic Qwen3-8B (k=8)"),
          ("25 groups", "Eedi level-2 topic", "Semantic MiniLM (k=25, = level-2 topic)", "Semantic Qwen3-8B (k=25, = level-2 topic)"),
          ("56 groups", "Eedi most specific tag", "Semantic MiniLM (k=56, = most specific tag)", "Semantic Qwen3-8B (k=56, = most specific tag)")]
SER = [("Topic tags", "#9a9a9a"), ("Semantic MiniLM", "#2a78d6"), ("Semantic Qwen3-8B", "#eb6834")]

fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.5), sharey=True)
for ax, retr in zip(axes, ["MiniLM", "Qwen3-0.6B"]):
    r = R[R.retriever == retr].set_index("partition"); x = np.arange(len(GROUPS)); w = 0.26
    for j, (lab, col) in enumerate(SER):
        vals = [r.loc[g[1 + j], "coverage"] for g in GROUPS]
        ax.bar(x + (j - 1) * w, vals, w * 0.92, color=col, label=lab, zorder=3)
        for xi, v in zip(x, vals): ax.text(xi + (j - 1) * w, v + 0.004, f"{v*100:.1f}", ha="center", va="bottom", fontsize=7)
        if j > 0:  # within-topic null (same topic mix per cluster)
            nl = [r.loc[g[1 + j], "null_within_L2"] for g in GROUPS]
            ax.scatter(x + (j - 1) * w, nl, marker="_", s=160, color="black", lw=1.4, zorder=4, label="Within-topic null" if j == 1 else None)
    free = [r.loc[g[1], "null_free"] for g in GROUPS]
    ax.scatter(x - w, free, marker="_", s=160, color="0.35", lw=1.0, ls="--", zorder=4, label="Random grouping" if retr == "MiniLM" else None)
    ax.set_xticks(x); ax.set_xticklabels([g[0] for g in GROUPS]); ax.set_ylim(0.5, 0.95)
    ax.set_title(f"Retriever: {retr}", fontsize=9.5); ax.grid(axis="y", color="0.92", lw=0.5); ax.set_axisbelow(True)
axes[0].set_ylabel("Dominant misconception-family share")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.01), fontsize=8)
fig.tight_layout(rect=(0, 0.08, 1, 1)); fig.savefig(FIG / "topic_baseline.png", dpi=300); plt.close(fig)
print("ok")
