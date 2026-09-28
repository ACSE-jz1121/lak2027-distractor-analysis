"""Who makes which error: misconception group x learner proficiency (Qwen3-Embedding-8B candidate groups, k = 8).

Learner proficiency = accuracy on all OTHER questions (leave-one-question-out), learners with >= 20 answers,
standardised over learners (z). For every distractor (one per question, 899) and every learner who attempted the question:
  sel      = 1 if the learner chose this distractor
Two curves / slopes per distractor:
  (a) P(sel | attempt)        : who makes this error at all (includes the general fact that weaker learners err more)
  (b) P(sel | wrong answer)   : among learners who got the item wrong, does this particular error rise or fall with
                                proficiency? (conditions on being wrong, so item difficulty and general error rate drop out)
Per-distractor logistic regressions sel ~ 1 + z (item-specific intercept absorbs item difficulty); log-odds ratio per SD.
Group level: mean log-OR over distractors (bootstrap CI), persistence ratio P(sel | top quarter) / P(sel | bottom quarter),
and an OLS of the per-distractor log-OR on group + item difficulty + base selection rate (difficulty-adjusted group means,
permutation test for the group effect).
Outputs -> results/proficiency_*.csv, figures/proficiency_by_group.png
"""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (EDM, TRAIN_CSV, META, IMAGES, DISTRACTORS, RESULTS, FIGURES, OUTPUTS, ASSETS, ROOT,
                    SEED as CFG_SEED, GROUP_ENCODER, GROUP_K, emb)
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

df = pd.read_parquet(DISTRACTORS).reset_index(drop=True)
grp = pd.read_csv(OUTPUTS / "selector_quartiles.csv").candidate.to_numpy()           # Qwen3-8B k = 8 candidate of each distractor
TIT = json.load(open(ASSETS / "titles.json", encoding="utf-8"))
AUD = {c["id"]: c["audit"]["verdict"] for c in json.load(open(OUTPUTS / "audit.json", encoding="utf-8"))["candidates"]}
rng = np.random.default_rng(42)

# ---------------- responses and leave-one-out proficiency
r = pd.read_csv(TRAIN_CSV, usecols=["QuestionId", "UserId", "AnswerValue", "IsCorrect"]).astype(int)
u = r.groupby("UserId").IsCorrect.agg(["sum", "size"]); u = u[u["size"] >= 20]
r = r[r.UserId.isin(u.index)].merge(u, left_on="UserId", right_index=True)
r["acc"] = (r["sum"] - r.IsCorrect) / (r["size"] - 1)                          # accuracy on the other questions
ua = u["sum"] / u["size"]; MU, SD = ua.mean(), ua.std()
r["z"] = (r.acc - MU) / SD
QCUT = np.quantile(ua, [0.25, 0.5, 0.75]); DCUT = np.quantile(ua, np.linspace(0.1, 0.9, 9))
r["q"] = np.searchsorted(QCUT, r.acc, side="right"); r["dec"] = np.searchsorted(DCUT, r.acc, side="right")
key = df[["QuestionId", "AnswerValue"]].rename(columns={"AnswerValue": "D"}).assign(i=np.arange(len(df)))
r = r.merge(key, on="QuestionId"); r["sel"] = (r.AnswerValue == r.D).astype(int)
print(f"responses {len(r):,}  learners {r.UserId.nunique():,}  distractors {r.i.nunique()}", flush=True)

def logit_fit(x, y, iters=30):
    """Two-parameter logistic regression by IRLS; returns slope, SE of slope."""
    X = np.column_stack([np.ones_like(x), x]); b = np.zeros(2)
    p0 = np.clip(y.mean(), 1e-4, 1 - 1e-4); b[0] = np.log(p0 / (1 - p0))
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(X @ b))); w = p * (1 - p) + 1e-12
        H = X.T @ (X * w[:, None]); g = X.T @ (y - p)
        step = np.linalg.solve(H, g); b += step
        if np.abs(step).max() < 1e-8: break
    cov = np.linalg.inv(H); return b[1], np.sqrt(cov[1, 1])

rows, curves = [], []
for i, g in r.groupby("i"):
    x, y = g.z.to_numpy(), g.sel.to_numpy().astype(float)
    b_all, se_all = logit_fit(x, y)
    w = g[g.IsCorrect == 0]; b_wr, se_wr = logit_fit(w.z.to_numpy(), w.sel.to_numpy().astype(float))
    pq = g.groupby("q").sel.mean().reindex(range(4)); pwq = w.groupby("q").sel.mean().reindex(range(4))
    rows.append(dict(i=i, sample=f"{int(df.QuestionId[i])}_row{i}", group=grp[i], n=len(g), n_wrong=len(w),
                     b_all=b_all, se_all=se_all, b_wrong=b_wr, se_wrong=se_wr,
                     p_q1=pq[0], p_q4=pq[3], pw_q1=pwq[0], pw_q4=pwq[3],
                     difficulty=float(df.difficulty[i]), selection=float(df.p_wrong_option[i])))
    for d, gd in g.groupby("dec"):
        wd = gd[gd.IsCorrect == 0]
        curves.append(dict(i=i, group=grp[i], dec=d, p_sel=gd.sel.mean(), p_sel_wrong=wd.sel.mean() if len(wd) else np.nan, n=len(gd)))
T = pd.DataFrame(rows); C = pd.DataFrame(curves)
T["persist"] = T.p_q4 / T.p_q1                                                       # >1: strongest quarter chooses it more often
T.to_csv(RESULTS / "proficiency_distractor_slopes.csv", index=False)

# ---------------- group summaries
def boot_ci(v, B=2000):
    v = v[~np.isnan(v)]; m = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(B)]
    return float(v.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))
# difficulty-adjusted group means: OLS  b ~ group + difficulty + logit(selection rate)
G = sorted(T.group.unique())
def adjusted(col):
    y = T[col].to_numpy(); ok = ~np.isnan(y)
    Xg = np.column_stack([(T.group == g).to_numpy().astype(float) for g in G])
    cov = np.column_stack([T.difficulty - T.difficulty.mean(),
                           np.log(T.selection / (1 - T.selection)) - np.log(T.selection / (1 - T.selection)).mean()])
    X = np.column_stack([Xg, cov])[ok]; yy = y[ok]
    beta, *_ = np.linalg.lstsq(X, yy, rcond=None); res = yy - X @ beta
    def F(lab):
        Xp = np.column_stack([np.column_stack([(lab == g).astype(float) for g in G]), cov[ok]])
        bp, *_ = np.linalg.lstsq(Xp, yy, rcond=None); rp = yy - Xp @ bp
        b0, *_ = np.linalg.lstsq(np.column_stack([np.ones(ok.sum()), cov[ok]]), yy, rcond=None); r0 = yy - np.column_stack([np.ones(ok.sum()), cov[ok]]) @ b0
        return ((r0 @ r0 - rp @ rp) / (len(G) - 1)) / ((rp @ rp) / (ok.sum() - len(G) - 2))
    lab = T.group.to_numpy()[ok]; f_obs = F(lab)
    f_null = np.array([F(rng.permutation(lab)) for _ in range(2000)])
    return dict(zip(G, beta[:len(G)])), float(f_obs), float((f_null >= f_obs).mean()), dict(zip(["difficulty", "logit_selection"], beta[len(G):]))
adj_all, F_all, p_all, cov_all = adjusted("b_all"); adj_wr, F_wr, p_wr, cov_wr = adjusted("b_wrong")
S = []
for g in G:
    t = T[T.group == g]
    m1, l1, h1 = boot_ci(t.b_all.to_numpy()); m2, l2, h2 = boot_ci(t.b_wrong.to_numpy())
    S.append(dict(group=g, title=TIT[g][0], verdict=AUD[g], n=len(t),
                  or_all=np.exp(m1), or_all_lo=np.exp(l1), or_all_hi=np.exp(h1), or_all_adj=np.exp(adj_all[g]),
                  or_wrong=np.exp(m2), or_wrong_lo=np.exp(l2), or_wrong_hi=np.exp(h2), or_wrong_adj=np.exp(adj_wr[g]),
                  p_q1=t.p_q1.mean(), p_q4=t.p_q4.mean(), persist_median=t.persist.median(),
                  share_rising_when_wrong=float((t.b_wrong > 0).mean())))
S = pd.DataFrame(S)
S.to_csv(RESULTS / "proficiency_group_summary.csv", index=False)
json.dump(dict(F_attempt=F_all, p_attempt=p_all, cov_attempt=cov_all, F_wrong=F_wr, p_wrong=p_wr, cov_wrong=cov_wr,
               learners=int(r.UserId.nunique()), responses=int(len(r)), quartile_cuts=QCUT.tolist(), mean_acc=MU, sd_acc=SD),
          open(RESULTS / "proficiency_group_tests.json", "w"), indent=1)
CG = C.groupby(["group", "dec"]).agg(p_sel=("p_sel", "mean"), p_sel_wrong=("p_sel_wrong", "mean")).reset_index()
CG.to_csv(RESULTS / "proficiency_curves.csv", index=False)

pd.set_option("display.width", 220)
print(S.drop(columns=["title"]).round(3).to_string(index=False))
print(f"group effect after difficulty adjustment: attempt F={F_all:.2f} p={p_all:.4f}; wrong-conditional F={F_wr:.2f} p={p_wr:.4f}")
print("covariates", cov_all, cov_wr)

# ---------------- figure
COL = {"S0": "#2a78d6", "S1": "#e8590c", "S4": "#1baf7a", "S5": "#7048e8", "S6": "#d6336c", "S7": "#b08900", "S2": "#9a9a9a", "S3": "#6d6d6d"}
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.0), gridspec_kw=dict(width_ratios=[1, 1, 1.05]))
x = np.arange(1, 11)
for g in G:
    cg = CG[CG.group == g].sort_values("dec"); ls = "--" if AUD[g].startswith("Not") else "-"
    lab = f"{g} {TIT[g][0]}" if not AUD[g].startswith("Not") else f"{g} (not coherent)"
    axes[0].plot(x, cg.p_sel, ls, color=COL[g], lw=1.6, marker="o", ms=3, label=lab)
    axes[1].plot(x, cg.p_sel_wrong, ls, color=COL[g], lw=1.6, marker="o", ms=3)
axes[0].set_title("(a) Chose the distractor, all attempts", fontsize=9.5, loc="left")
axes[1].set_title("(b) Chose the distractor, among wrong answers", fontsize=9.5, loc="left")
for ax in axes[:2]:
    ax.set_xlabel("Learner proficiency decile (1 = weakest)"); ax.set_xticks(x); ax.grid(color="0.92", lw=0.5); ax.set_axisbelow(True)
axes[0].set_ylabel("Mean selection probability")
ax = axes[2]; order = S.sort_values("or_wrong_adj").group.tolist()
for k, g in enumerate(order):
    s = S[S.group == g].iloc[0]
    ax.errorbar(s.or_all, k + 0.15, xerr=[[s.or_all - s.or_all_lo], [s.or_all_hi - s.or_all]], fmt="o", color="#555555", ms=4, capsize=2, lw=1)
    ax.errorbar(s.or_wrong, k - 0.15, xerr=[[s.or_wrong - s.or_wrong_lo], [s.or_wrong_hi - s.or_wrong]], fmt="s", color=COL[g], ms=4.5, capsize=2, lw=1.2)
ax.axvline(1, color="0.5", lw=0.8, ls=":")
ax.set_yticks(range(len(order))); ax.set_yticklabels([f"{g}" for g in order]); ax.set_xscale("log")
ax.set_xlabel("Odds ratio per SD of proficiency (log scale)")
ax.set_title("(c) Change in odds per SD of proficiency", fontsize=9.5, loc="left")
ax.plot([], [], "o", color="#555555", label="all attempts"); ax.plot([], [], "s", color="#333333", label="among wrong answers")
ax.legend(frameon=False, fontsize=7.5, loc="center")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=8, bbox_to_anchor=(0.45, -0.1))
fig.tight_layout(rect=(0, 0.08, 1, 1)); fig.savefig(FIGURES / "proficiency_by_group.png", dpi=300, bbox_inches="tight"); print("figure saved")
