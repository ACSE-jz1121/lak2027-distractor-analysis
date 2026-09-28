"""Encoder-comparison figures (02_encoder_comparison.py) -> figures/encoders_*.png"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import (EDM, TRAIN_CSV, META, IMAGES, DISTRACTORS, MISC_ZIP, RESULTS, FIGURES, OUTPUTS, ASSETS,
                    SEED as CFG_SEED, GROUP_ENCODER, GROUP_K, emb)
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

FIG = FIGURES
S = pd.read_csv(RESULTS / "encoder_comparison.csv"); G = pd.read_csv(RESULTS / "encoder_tuning_grid.csv"); P = pd.read_csv(RESULTS / "encoder_pairwise_ari_k8.csv", index_col=0)
SHORT = {"SciBERT": "SciBERT", "all-MiniLM-L6-v2": "MiniLM-L6", "Qwen3-Embedding-0.6B": "Qwen3-0.6B",
         "Qwen3-Embedding-4B": "Qwen3-4B", "Qwen3-Embedding-8B": "Qwen3-8B", "bge-large-en-v1.5": "bge-large",
         "e5-large-v2": "e5-large", "gte-large-en-v1.5": "gte-large", "mxbai-embed-large-v1": "mxbai-large",
         "nomic-embed-text-v1.5": "nomic-v1.5", "all-mpnet-base-v2": "mpnet-base"}
S["short"] = S.encoder.map(SHORT).fillna(S.encoder)
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False})
HL = {"MiniLM-L6": "#2a78d6", "SciBERT": "#eb6834", "Qwen3-8B": "#1baf7a"}

# ---- Fig 1: stability vs external alignment at fixed k = 8
fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
for ax, (y, yl) in zip(axes, [("k8_align_Qwen3-0.6B_all", "Dominant misconception-family share\n(all instances, Qwen3 retriever)"),
                              ("k8_NMI_leaf", "NMI with Eedi subject tag")]):
    for _, r in S.iterrows():
        c = HL.get(r.short, "#555555")
        ax.scatter(r.k8_stability, r[y], s=36, color=c, edgecolor="white", lw=0.8, zorder=3)
        ax.annotate(r.short, (r.k8_stability, r[y]), xytext=(4, 3), textcoords="offset points", fontsize=8, color="0.2")
    ax.set_xlabel("Bootstrap stability at k = 8 (mean ARI, training split)"); ax.set_ylabel(yl)
    ax.grid(color="0.92", lw=0.5); ax.set_axisbelow(True)
if "k8_align_Qwen3-0.6B_all_null" in S:
    axes[0].axhline(S["k8_align_Qwen3-0.6B_all_null"].mean(), color="0.5", lw=0.8, ls="--")
    axes[0].annotate("permutation baseline", (axes[0].get_xlim()[0], S["k8_align_Qwen3-0.6B_all_null"].mean()),
                     xytext=(3, 3), textcoords="offset points", fontsize=7.5, color="0.4")
fig.tight_layout(); fig.savefig(FIG / "encoders_stability_vs_alignment_k8.png", dpi=300); plt.close(fig)

# ---- Fig 2: stability by k (best d per k) for each encoder
G = G[G.encoder.isin(S.encoder)]
best = G.groupby(["encoder", "k"]).Mean_ARI.max().reset_index()
fig, ax = plt.subplots(figsize=(6.4, 3.9))
for e, g in best.groupby("encoder"):
    s = SHORT.get(e, e); c = HL.get(s, None)
    ax.plot(g.k, g.Mean_ARI, marker="o", ms=4, lw=2 if c else 1, color=c if c else None, alpha=1 if c else 0.6, label=s)
ax.set_xticks(sorted(best.k.unique())); ax.set_xlabel("Number of clusters k"); ax.set_ylabel("Bootstrap stability (best PCA setting)")
ax.grid(color="0.92", lw=0.5); ax.set_axisbelow(True); ax.legend(fontsize=7.5, frameon=False, ncol=2, loc="upper right")
fig.tight_layout(); fig.savefig(FIG / "encoders_stability_by_k.png", dpi=300); plt.close(fig)

# ---- Fig 3: agreement between encoders' k = 8 partitions
keep = [e for e in P.index if e in set(S.encoder)]; P = P.loc[keep, keep]; lab = [SHORT.get(e, e) for e in keep]
fig, ax = plt.subplots(figsize=(6.2, 5.2))
im = ax.imshow(P.values, cmap="Blues", vmin=0, vmax=1)
ax.set_xticks(range(len(lab))); ax.set_xticklabels(lab, rotation=45, ha="right"); ax.set_yticks(range(len(lab))); ax.set_yticklabels(lab)
for i in range(len(lab)):
    for j in range(len(lab)):
        v = P.values[i, j]; ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7, color="white" if v > 0.6 else "0.15")
ax.spines[:].set_visible(False); fig.colorbar(im, ax=ax, fraction=0.04, label="ARI between k = 8 partitions")
fig.tight_layout(); fig.savefig(FIG / "encoders_pairwise_ari_k8.png", dpi=300); plt.close(fig)
print("figs ok")
