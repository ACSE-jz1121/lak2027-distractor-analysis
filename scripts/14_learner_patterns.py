"""Learner-pattern summary across the eight semantic candidate groups (figure + table + PDF).

Inputs (no new analysis; everything comes from existing outputs):
  results/who_makes_each_error.csv      selection by proficiency quartile, OR_adj, recurrence, confidence (step 12)
  results/group_behaviour.csv           number of distractors with confidence ratings (step 9)
  scripts/13_review_workspace.py        TITLES / PROFILE / INDICATE (same wording as the interface)
Outputs:
  figures/learner_patterns.png, figures/learner_patterns.pdf
  results/learner_patterns_table.tex, results/learner_patterns_interpretation_table.tex
"""
import ast
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import RESULTS as RES, FIGURES as FIG
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt



def gui_dicts():
    """Read TITLES/PROFILE/INDICATE from the interface builder so the wording stays identical."""
    tree = ast.parse((Path(__file__).resolve().parent / "13_review_workspace.py").read_text(encoding="utf-8"))
    out = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id in ("TITLES", "PROFILE", "INDICATE"):
            out[node.targets[0].id] = ast.literal_eval(node.value)
    return out


D = gui_dicts()
W = pd.read_csv(RES / "who_makes_each_error.csv")
GB = pd.read_csv(RES / "group_behaviour.csv").set_index("group")
W["conf_rated"] = W.group.map(GB.conf_rated)
W["n"] = W.group.map(GB.n)
STATUS = {"S": "Supported", "P": "Partly supported", "N": "Not coherent"}
W["status_full"] = W.status.map(STATUS)
ORDER = ["S7", "S0", "S6", "S5", "S4", "S1", "S2", "S3"]          # supported (by recurrence), partial, not coherent
W = W.set_index("group").loc[ORDER].reset_index()


# ---- verbal levels: identical rules to the GUI (make_review_workspace.py / template)
def prof_head(r):
    if r.chosen_q4 >= 0.16: return "Still fairly common among stronger learners"
    return "Steep decline" if r.or_adj <= 0.60 else "Declines" if r.or_adj <= 0.66 else "Gradual decline"


def rec_head(r):
    if not r.spec_all_lo > 0: return "Little evidence"
    return "Yes" if r.spec_wrong_lo > 0 else "Partly"


def conf_head(r):
    if pd.isna(r.confident_error): return "Too few ratings"
    c = r.confident_error
    return "Often" if c >= 0.30 else "Sometimes" if c >= 0.18 else "Usually not"


W["prof_head"] = W.apply(prof_head, axis=1)
W["rec_head"] = W.apply(rec_head, axis=1)
W["conf_head"] = W.apply(conf_head, axis=1)

# ---- figure: three aligned panels, one row per group (answers the GUI's three questions)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
INK, MUTED, LINE = "#1b1f24", "#6a7280", "#d5d9df"
C_Q1, C_Q4, C_ALL, C_WR, C_CONF = "#2f5d8c", "#9db9d6", "#2f5d8c", "#c2703d", "#5b7f6a"
labels = [f"{g}  {s}" for g, s in zip(W.group, ["Factors vs multiples", "Negative-number signs", "Angle estimation",
                                                 "Coordinates/transformation", "Area and perimeter", "Fraction representation",
                                                 "Mixed shapes/symmetry (not coherent)", "Mixed arithmetic (not coherent)"])]
y = np.arange(len(W))[::-1]
fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.9), sharey=True, gridspec_kw=dict(width_ratios=[1.15, 1.2, 0.85], wspace=0.12))

ax = axes[0]
for yi, r in zip(y, W.itertuples()):
    ax.plot([r.chosen_q4, r.chosen_q1], [yi, yi], color=LINE, lw=2.2, zorder=1, solid_capstyle="round")
ax.scatter(W.chosen_q1, y, s=34, color=C_Q1, zorder=3, label="Weakest quarter")
ax.scatter(W.chosen_q4, y, s=34, color="white", edgecolor=C_Q1, linewidth=1.5, zorder=3, label="Strongest quarter")
for yi, r in zip(y, W.itertuples()):
    ax.text(0.345, yi, f"OR {r.or_adj:.2f}", va="center", ha="left", fontsize=7.8, color=MUTED)
ax.set_xlim(0.08, 0.40); ax.set_xticks([0.1, 0.2, 0.3]); ax.set_xticklabels(["10%", "20%", "30%"])
ax.set_title("How does it change with proficiency?", loc="left", fontsize=9.5, color=INK, fontweight="bold")
ax.set_xlabel("Share of attempts selecting the distractor")
ax.legend(loc="lower center", bbox_to_anchor=(0.45, -0.30), ncol=2, frameon=False, fontsize=8, handletextpad=0.3)

ax = axes[1]
ax.axvline(0, color=LINE, lw=1, zorder=0)
off = 0.14
for yi, r in zip(y, W.itertuples()):
    ax.plot([r.spec_all_lo, r.spec_all_hi], [yi + off] * 2, color=C_ALL, lw=1.3, alpha=.8)
    ax.plot([r.spec_wrong_lo, r.spec_wrong_hi], [yi - off] * 2, color=C_WR, lw=1.3, alpha=.8)
ax.scatter(W.spec_all, y + off, s=30, color=C_ALL, zorder=3, label="All attempts")
ax.scatter(W.spec_wrong, y - off, s=30, marker="s", color=C_WR, zorder=3, label="Incorrect responses only")
ax.set_xlim(-0.10, 0.17); ax.set_xticks([-0.05, 0, 0.05, 0.10, 0.15])
ax.set_title("Does it recur in the same learners?", loc="left", fontsize=9.5, color=INK, fontweight="bold")
ax.set_xlabel("Learner-specific recurrence (95% CI)")
ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.30), ncol=2, frameon=False, fontsize=8, handletextpad=0.3)

ax = axes[2]
conf = W.confident_error.fillna(0)
ax.barh(y, conf, height=0.55, color=C_CONF, alpha=.85)
for yi, r in zip(y, W.itertuples()):
    txt = "too few ratings" if pd.isna(r.confident_error) else f"{100 * r.confident_error:.0f}%"
    ax.text((0 if pd.isna(r.confident_error) else r.confident_error) + 0.01, yi, txt, va="center", fontsize=8, color=INK if not pd.isna(r.confident_error) else MUTED)
ax.set_xlim(0, 0.6); ax.set_xticks([0, 0.2, 0.4]); ax.set_xticklabels(["0%", "20%", "40%"])
ax.set_title("Confident when wrong?", loc="left", fontsize=9.5, color=INK, fontweight="bold")
ax.set_xlabel("Confident errors")

axes[0].set_yticks(y); axes[0].set_yticklabels(labels)
for a in axes:
    a.tick_params(axis="y", length=0); a.grid(axis="x", color="#eceef1", lw=0.8); a.set_axisbelow(True)
# separators between status blocks (supported | partial | not coherent)
for a in axes:
    for b in (2.5, 1.5):
        a.axhline(b, color="#aeb4bd", lw=0.7, ls=(0, (3, 3)))
fig.savefig(FIG / "learner_patterns.png", dpi=300, bbox_inches="tight")
fig.savefig(FIG / "learner_patterns.pdf", bbox_inches="tight")
plt.close(fig)


# ---- table (English, LaTeX)
def tex(s):
    return (s.replace("&", r"\&").replace("%", r"\%").replace("—", "---").replace("–", "--"))


def star(m, lo):
    return f"{m:+.2f}" + (r"$^{*}$" if lo > 0 else "")


rows, irows = [], []
for r in W.itertuples():
    conf = f"({int(r.conf_rated)} of {int(r.n)} rated)" if pd.isna(r.confident_error) else f"{100 * r.confident_error:.0f}\\%"
    rows.append(
        f"{r.group} & {tex(D['TITLES'][r.group])}\\newline\\textit{{{r.status_full}}} & {tex(D['PROFILE'][r.group][0])} & "
        f"{tex(r.prof_head)}\\newline {100 * r.chosen_q1:.0f}\\% $\\rightarrow$ {100 * r.chosen_q4:.0f}\\%; OR {r.or_adj:.2f} & "
        f"{tex(r.rec_head)}\\newline {star(r.spec_all, r.spec_all_lo)} / {star(r.spec_wrong, r.spec_wrong_lo)} & "
        f"{tex(r.conf_head)}\\newline {conf} \\\\")
    irows.append(f"{r.group} & {tex(D['TITLES'][r.group])} & {tex(D['INDICATE'][r.group])} \\\\")


def blocks(rs):
    return "\n".join(rs[:5]) + "\n\\midrule\n" + rs[5] + "\n\\midrule\n" + "\n".join(rs[6:])


table = r"""\begin{table*}[t]
\centering
\caption{How each candidate error appears across learners. Change with proficiency: share of attempts selecting the group's distractors in the weakest $\rightarrow$ strongest proficiency quarter, and the difficulty-adjusted odds ratio (OR) per SD of proficiency. Recurrence: learner-specific split-half recurrence for all attempts / incorrect responses only ($^{*}$95\% CI excludes 0). Confidence: share of rated distractors whose selectors were about as confident as correct responders. Groups are ordered supported, partly supported, not coherent. The summaries describe response patterns across learners, not individual learners' misconceptions.}
\label{tab:learner-patterns}
\small
\renewcommand{\arraystretch}{1.3}
\begin{tabularx}{\textwidth}{@{}l >{\raggedright\arraybackslash}X >{\raggedright\arraybackslash}p{3.6cm} >{\raggedright\arraybackslash}p{3.6cm} >{\raggedright\arraybackslash}p{2.7cm} >{\raggedright\arraybackslash}p{2.5cm}@{}}
\toprule
 & Candidate error & Learner pattern & Change with proficiency & Recurs in the same learners? & Confident when wrong? \\
\midrule
""" + blocks(rows) + r"""
\bottomrule
\end{tabularx}
\end{table*}
"""
itable = r"""\begin{table*}[t]
\centering
\caption{Possible interpretation of each learner pattern, as shown in the review interface. These readings are observational and do not imply that any instructional intervention would be effective.}
\label{tab:learner-pattern-interpretation}
\small
\renewcommand{\arraystretch}{1.3}
\begin{tabularx}{\textwidth}{@{}l >{\raggedright\arraybackslash}p{5.2cm} >{\raggedright\arraybackslash}X@{}}
\toprule
 & Candidate error & Possible interpretation \\
\midrule
""" + blocks(irows) + r"""
\bottomrule
\end{tabularx}
\end{table*}
"""
(RES / "learner_patterns_interpretation_table.tex").write_text(itable, encoding="utf-8")
(RES / "learner_patterns_table.tex").write_text(table, encoding="utf-8")
print(W[["group", "prof_head", "rec_head", "conf_head"]].to_string(index=False))
