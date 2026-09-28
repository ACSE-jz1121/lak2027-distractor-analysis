"""Figures for the semantic-behavioural weight path (secondary analysis).
  main : one panel, Qwen3-Embedding-8B    -> figures/weight_path_main.png
  supp : two panels, MiniLM and SciBERT   -> figures/weight_path_supp.png
Reads results/weight_path.csv (path B: semantic block without PCA, unit-variance blocks)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import (EDM, TRAIN_CSV, META, IMAGES, DISTRACTORS, MISC_ZIP, RESULTS, FIGURES, OUTPUTS, ASSETS,
                    SEED as CFG_SEED, GROUP_ENCODER, GROUP_K, emb)
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = FIGURES
wp = pd.read_csv(RESULTS / "weight_path.csv")
wp = wp[wp.path.str.startswith("B")]

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 9, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "legend.frameon": False,
})
# colour + marker + line style, so the figure also reads in greyscale
SERIES = [("bootstrap_ari",             "Bootstrap stability",                 "#2a78d6", "o", "-"),
          ("ari_vs_semantic",           "Similarity to semantic partition",    "#eb6834", "s", "--"),
          ("ari_vs_behavioural_same_k", "Similarity to behavioural partition", "#1baf7a", "^", "-.")]

def draw(ax, enc, title, ylabel=True):
    g = wp[wp.encoder == enc].sort_values("behavioural_share")
    for col, lab, c, m, ls in SERIES:
        ax.plot(g.behavioural_share, g[col], ls, color=c, marker=m, ms=3.6, lw=1.3, label=lab,
                markeredgewidth=0, clip_on=False, zorder=3)
    ax.axvline(0.971, color="0.35", lw=0.7, ls=(0, (3, 2)), zorder=1)
    ax.text(0.958, 0.42, "Selected by silhouette", rotation=90, ha="right", va="center", fontsize=7, color="0.3")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.0)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1]); ax.set_xticklabels(["0", "0.25", "0.50", "0.75", "1"])
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    # endpoint labels between the tick labels and the axis title
    for x, txt, ha in [(0, "Semantic-only", "left"), (1, "Behavioural-only", "right")]:
        ax.annotate(txt, xy=(x, 0), xycoords=("data", "axes fraction"), xytext=(0, -17), textcoords="offset points",
                    ha=ha, va="top", fontsize=8.5)
    ax.set_xlabel("Behavioural share of total feature variance", labelpad=14)
    if ylabel: ax.set_ylabel("ARI")
    ax.set_title(title, fontsize=9.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="0.9", lw=0.5); ax.set_axisbelow(True)

# ---------------- main: primary encoder
fig, ax = plt.subplots(figsize=(4.8, 3.3))
draw(ax, "Qwen3-Embedding-8B", "Semantic–behavioural weight path (Qwen3-Embedding-8B)")
h, l = ax.get_legend_handles_labels()
ax.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, -0.36), ncol=2, fontsize=7.8, handlelength=2.6, columnspacing=1.4)
fig.tight_layout()
fig.savefig(OUT / "weight_path_main.png", dpi=600, bbox_inches="tight")
plt.close(fig)

# ---------------- supplementary: MiniLM and SciBERT
fig, axes = plt.subplots(1, 2, figsize=(6.9, 3.1), sharey=True)
draw(axes[0], "MiniLM", "(a) Semantic–behavioural weight path (MiniLM)")
draw(axes[1], "SciBERT", "(b) Semantic–behavioural weight path (SciBERT)", ylabel=False)
for a in axes: a.title.set_fontsize(8.5)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=3, fontsize=8, bbox_to_anchor=(0.5, -0.04), handlelength=2.6)
fig.tight_layout(rect=(0, 0.07, 1, 1), w_pad=2.0)
fig.savefig(OUT / "weight_path_supp.png", dpi=600, bbox_inches="tight")
print("saved figures/weight_path_main.png and figures/weight_path_supp.png")
