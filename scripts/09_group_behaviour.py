"""Group-level behavioural table for the Qwen3-Embedding-8B semantic candidate groups (k = 8).
Per group: median selection rate, error rate, selector proficiency and proficiency gap; ability-quarter mix of selectors
(weighted by number of selectors) against the mix among attempters; confidence (distractors with >= 20 ratings);
dominant 2x2 response profile. Across groups: Kruskal-Wallis test and eta^2 (rank-based) per feature.
Outputs -> results/group_behaviour.csv, results/group_behaviour_tests.csv, results/group_behaviour_table.tex"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (EDM, TRAIN_CSV, META, IMAGES, DISTRACTORS, MISC_ZIP, RESULTS, FIGURES, OUTPUTS, ASSETS,
                    SEED as CFG_SEED, GROUP_ENCODER, GROUP_K, emb)
import json
import numpy as np, pandas as pd
from scipy.stats import kruskal, rankdata

df = pd.read_parquet(DISTRACTORS).reset_index(drop=True)
A = json.load(open(OUTPUTS / "audit.json", encoding="utf-8")); TIT = json.load(open(ASSETS / "titles.json", encoding="utf-8"))
QQ = pd.read_csv(OUTPUTS / "selector_quartiles.csv"); CF = pd.read_csv(OUTPUTS / "confidence.csv")
assert (QQ["sample"] == CF["sample"]).all()
F = pd.DataFrame({"group": QQ.candidate, "selection": df.p_wrong_option, "error": df.difficulty,
                  "selector_prof": df.avg_student_acc_wrong, "gap": df.discrimination_proxy})
VZ = {"Supported candidate": "Supported", "Partly supported": "Partly supported", "Not a coherent misconception candidate": "Not coherent"}

def row(mask, gid, title, verdict):
    q = QQ[mask]; w = q.n_selectors.to_numpy(); c = CF[mask]; ok = c[c.enough]
    sel = [float((w * q[f"share_Q{k}"]).sum() / w.sum()) for k in (1, 4)]
    att = [float((w * q[f"att_Q{k}"]).sum() / w.sum()) for k in (1, 4)]
    f = F[mask]
    return dict(group=gid, title=title, verdict=verdict, n=int(mask.sum()),
                selection=f.selection.median(), error=f.error.median(), selector_prof=f.selector_prof.median(), gap=f.gap.median(),
                q1_selectors=sel[0], q1_attempters=att[0], q4_selectors=sel[1], q4_attempters=att[1],
                q4_overrep=float((q.lift_Q4 >= 1).mean()),
                conf_rated=int(len(ok)), conf_selectors=float(np.average(ok.sel_conf, weights=ok.n_conf)) if len(ok) else np.nan,
                conf_correct=float(np.average(ok.corr_conf, weights=ok.n_conf)) if len(ok) else np.nan,
                confident_error=float(ok.confident_error.mean()) if len(ok) else np.nan)
rows = []
for cnd in A["candidates"]:
    prof = max(cnd["profile_shares"].items(), key=lambda kv: kv[1])
    r = row((QQ.candidate == cnd["id"]).to_numpy(), cnd["id"], TIT[cnd["id"]][0], VZ[cnd["audit"]["verdict"]])
    r.update(profile=prof[0], profile_share=prof[1]); rows.append(r)
allr = row(np.ones(len(QQ), bool), "All", "All 899 distractors", ""); allr.update(profile="", profile_share=np.nan); rows.append(allr)
T = pd.DataFrame(rows); T.to_csv(RESULTS / "group_behaviour.csv", index=False)

tests = []
for f in ["selection", "error", "selector_prof", "gap"]:
    groups = [F.loc[F.group == g, f].dropna().to_numpy() for g in sorted(F.group.unique())]
    H, p = kruskal(*groups); n = sum(len(g) for g in groups); k = len(groups)
    tests.append(dict(feature=f, H=H, p=p, eta2=(H - k + 1) / (n - k)))
lift = QQ.lift_Q1.replace([np.inf, -np.inf], np.nan)
H, p = kruskal(*[lift[(QQ.candidate == g) & lift.notna()].to_numpy() for g in sorted(QQ.candidate.unique())])
tests.append(dict(feature="weakest-quarter lift", H=H, p=p, eta2=(H - 7) / (lift.notna().sum() - 8)))
Tt = pd.DataFrame(tests); Tt.to_csv(RESULTS / "group_behaviour_tests.csv", index=False)

# LaTeX table for the paper (English)
def p(x): return f"{100*x:.0f}\\%"
SHORT_EN = {"S0": "Negative-number sign errors", "S1": "Fraction representation", "S2": "Mixed shapes and symmetry$^\\dagger$",
            "S3": "Mixed arithmetic$^\\dagger$", "S4": "Area and perimeter", "S5": "Coordinates and transformations",
            "S6": "Angle size from diagram", "S7": "Factors vs.\\ multiples"}
lines = []
for _, r in T.iterrows():
    name = SHORT_EN.get(r.group, r.title) if r.group != "All" else "\\textit{All distractors}"
    conf = "--" if (np.isnan(r.conf_selectors) or r.conf_rated < 0.3 * r.n) else f"{r.conf_selectors:.0f} / {r.conf_correct:.0f} ({p(r.confident_error)})"
    lines.append(f"{r.group} & {name} & {r.n} & {p(r.selection)} & {p(r.error)} & {r.selector_prof:.2f} & {r.gap:.2f} & "
                 f"{p(r.q1_selectors)} ({p(r.q1_attempters)}) & {p(r.q4_selectors)} ({p(r.q4_attempters)}) & {conf} \\\\" + ("\\midrule" if r.group == "S7" else ""))
tex = r"""\begin{table*}[t]
\centering\small
\caption{Learner-response behaviour of the semantic candidate groups (Qwen3-Embedding-8B, $k=8$). Values are medians over the group's distractors, except the quarter shares, which are pooled over all selectors of the group (in brackets: the same share among all learners who attempted these questions). Confidence: mean self-rated confidence of selectors / correct responders over distractors with at least 20 ratings, and the share of those distractors where selectors are within 5 points of correct responders; -- = fewer than 30\% of the group's distractors rated. $^\dagger$Not coherent under the audit rule.}
\label{tab:group-behaviour}
\resizebox{\textwidth}{!}{\begin{tabular}{llrrrrrccc}
\toprule
 & Candidate & $n$ & Selected & Error rate & Selector acc. & Gap & Weakest quarter & Strongest quarter & Confidence \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}}
\end{table*}
"""
(RESULTS / "group_behaviour_table.tex").write_text(tex, encoding="utf-8")
pd.set_option("display.width", 250)
print(T.drop(columns=["title"]).round(3).to_string(index=False)); print(Tt.round(4).to_string(index=False))
