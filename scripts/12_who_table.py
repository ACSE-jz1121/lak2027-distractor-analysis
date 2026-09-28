"""'Who makes each error' table: replaces the group-median table.
Per candidate group (Qwen3-Embedding-8B, k = 8): how common the error is overall and in the weakest / strongest quarter
of learners, how fast it fades with proficiency after adjusting for item difficulty, whether the same learners make it
repeatedly across quizzes (ability-adjusted split-half consistency beyond the cross-group level), whether those learners
also choose this specific distractor when wrong, and how often it is a confident error.
Inputs: results/proficiency_*.csv, results/learner_consistency.csv, results/group_behaviour.csv
Outputs: results/who_makes_each_error.csv, results/who_makes_each_error_table.tex (English, for the paper)"""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (EDM, TRAIN_CSV, META, IMAGES, DISTRACTORS, RESULTS, FIGURES, OUTPUTS, ASSETS, ROOT,
                    SEED as CFG_SEED, GROUP_ENCODER, GROUP_K, emb)
import numpy as np, pandas as pd

P = pd.read_csv(RESULTS / "proficiency_group_summary.csv").set_index("group")
T = pd.read_csv(RESULTS / "proficiency_distractor_slopes.csv")
L = pd.read_csv(RESULTS / "learner_consistency.csv"); La = L[L.version == "all"].set_index("group"); Lw = L[L.version == "wrong"].set_index("group")
B = pd.read_csv(RESULTS / "group_behaviour.csv").set_index("group")
SHORT = {"S0": "Negative-number sign errors", "S1": "Fraction representation", "S2": "Mixed shapes and symmetry",
         "S3": "Mixed arithmetic", "S4": "Area and perimeter", "S5": "Coordinates and transformations",
         "S6": "Angle size from diagram", "S7": "Factors vs.\\ multiples"}
STAT = {"Supported candidate": "S", "Partly supported": "P", "Not a coherent misconception candidate": "N"}
order = [g for g in P.index if not P.loc[g, "verdict"].startswith("Not")] + [g for g in P.index if P.loc[g, "verdict"].startswith("Not")]

rows = []
for g in order:
    conf = B.loc[g]
    rows.append(dict(group=g, candidate=SHORT[g], status=STAT[P.loc[g, "verdict"]],
                     chosen_all=T[T.group == g].selection.mean(), chosen_q1=P.loc[g, "p_q1"], chosen_q4=P.loc[g, "p_q4"],
                     or_adj=P.loc[g, "or_all_adj"],
                     spec_all=La.loc[g, "specificity"], spec_all_lo=La.loc[g, "spec_lo"], spec_all_hi=La.loc[g, "spec_hi"],
                     spec_wrong=Lw.loc[g, "specificity"], spec_wrong_lo=Lw.loc[g, "spec_lo"], spec_wrong_hi=Lw.loc[g, "spec_hi"],
                     confident_error=np.nan if conf.conf_rated < 0.3 * conf.n else conf.confident_error))
W = pd.DataFrame(rows); W.to_csv(RESULTS / "who_makes_each_error.csv", index=False)

def pc(x): return f"{100*x:.0f}\\%"
def spec(m, lo, hi):
    star = "$^{*}$" if lo > 0 else ""
    return f"{m:+.2f}{star}"
lines = []
for _, r in W.iterrows():
    ce = "--" if np.isnan(r.confident_error) else pc(r.confident_error)
    lines.append(f"{r.group} & {r.candidate} & {r.status} & {pc(r.chosen_all)} & {pc(r.chosen_q1)} & {pc(r.chosen_q4)} & "
                 f"{r.or_adj:.2f} & {spec(r.spec_all, r.spec_all_lo, r.spec_all_hi)} & {spec(r.spec_wrong, r.spec_wrong_lo, r.spec_wrong_hi)} & {ce} \\\\"
                 + ("\\midrule" if r.group == order[5] else ""))
tex = r"""\begin{table*}[t]
\caption{Who makes each error. \emph{Chosen}: mean share of attempts that choose the group's distractors, overall and among the weakest and strongest quarter of learners (accuracy on the other questions). \emph{Fades}: odds ratio per SD of learner proficiency, from per-distractor logistic regressions, adjusted for item difficulty (lower = disappears faster as proficiency rises). \emph{Same learners}: ability-adjusted split-half consistency across different quizzes, minus the mean consistency with the other groups; \emph{any error} uses all attempts, \emph{this error} only wrong answers (choosing this distractor rather than another wrong option). $^{*}$95\% bootstrap interval excludes zero. \emph{Confident}: share of distractors (with at least 20 ratings) whose selectors are as confident as correct responders; -- = too few ratings. Status: S supported, P partly supported, N not coherent.}
\label{tab:who}
\centering\small
\resizebox{\textwidth}{!}{\begin{tabular}{llcrrrrrrr}
\toprule
 & & & \multicolumn{3}{c}{Chosen} & Fades & \multicolumn{2}{c}{Same learners} & \\
\cmidrule(lr){4-6}\cmidrule(lr){8-9}
 & Candidate & Status & All & Weakest & Strongest & OR/SD & Any error & This error & Confident \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}}
\end{table*}
"""
open(RESULTS / "who_makes_each_error_table.tex", "w", encoding="utf-8").write(tex)
pd.set_option("display.width", 220); print(W.round(3).to_string(index=False))
