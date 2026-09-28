"""Do the same learners make the same kind of error? Learner-level, ability-adjusted consistency per candidate group.

For every response on the 899 distractor items (learners with >= 20 answers):
  expected probability of choosing the item's distractor given the learner's proficiency (accuracy on all OTHER
  questions, standardised) from a per-item logistic regression (the item intercept absorbs difficulty);
  residual = chose (0/1) - expected.
Two versions:
  (A) all attempts          -> tendency to make this group's error beyond what ability predicts (includes topic weakness)
  (B) wrong answers only    -> when wrong, tendency to choose this group's distractor rather than another wrong option
                               (the misconception-specific choice; general topic weakness drops out)
Split-half design at the QUIZ level: each learner's quizzes are randomly assigned to two halves, so the two halves never
share a quiz/session. For learner u and group g, A_g = mean residual in half 1, B_g = mean residual in half 2
(>= 2 items per half). Within-group consistency r(A_g, B_g) is compared with cross-group r(A_g, B_h), h != g:
within > cross means that a learner's tendency is specific to that candidate misconception, not a general error tendency.
Bootstrap over learners (1,000) for CIs; 20 random quiz splits averaged.
Outputs -> results/learner_consistency*.csv|json, figures/learner_consistency.png
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
grp = pd.read_csv(OUTPUTS / "selector_quartiles.csv").candidate.to_numpy()
TIT = json.load(open(ASSETS / "titles.json", encoding="utf-8"))
AUD = {c["id"]: c["audit"]["verdict"] for c in json.load(open(OUTPUTS / "audit.json", encoding="utf-8"))["candidates"]}
G = sorted(set(grp)); NS, NB, MINH = 20, 1000, 2

r = pd.read_csv(TRAIN_CSV, usecols=["QuestionId", "UserId", "AnswerId", "AnswerValue", "IsCorrect"])
meta = pd.read_csv(META / "answer_metadata_task_3_4.csv", usecols=["AnswerId", "QuizId"])
r = r.merge(meta, on="AnswerId", how="left")
u = r.groupby("UserId").IsCorrect.agg(["sum", "size"]); u = u[u["size"] >= 20]
r = r[r.UserId.isin(u.index)].merge(u, left_on="UserId", right_index=True)
r["acc"] = (r["sum"] - r.IsCorrect) / (r["size"] - 1); ua = u["sum"] / u["size"]
r["z"] = (r.acc - ua.mean()) / ua.std()
key = df[["QuestionId", "AnswerValue"]].rename(columns={"AnswerValue": "D"}).assign(i=np.arange(len(df)), g=grp)
r = r.merge(key, on="QuestionId"); r["sel"] = (r.AnswerValue == r.D).astype(float)
print(f"responses {len(r):,} learners {r.UserId.nunique():,} quizzes {r.QuizId.nunique():,}", flush=True)

def fit(x, y, iters=30):
    X = np.column_stack([np.ones_like(x), x]); p0 = np.clip(y.mean(), 1e-4, 1 - 1e-4); b = np.array([np.log(p0 / (1 - p0)), 0.0])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(X @ b))); w = p * (1 - p) + 1e-12
        step = np.linalg.solve(X.T @ (X * w[:, None]), X.T @ (y - p)); b += step
        if np.abs(step).max() < 1e-8: break
    return b
r["res_all"] = np.nan; r["res_wrong"] = np.nan
for i, gi in r.groupby("i"):
    b = fit(gi.z.to_numpy(), gi.sel.to_numpy()); p = 1 / (1 + np.exp(-(b[0] + b[1] * gi.z.to_numpy())))
    r.loc[gi.index, "res_all"] = gi.sel.to_numpy() - p
    w = gi[gi.IsCorrect == 0]
    if len(w) >= 20:
        b = fit(w.z.to_numpy(), w.sel.to_numpy()); p = 1 / (1 + np.exp(-(b[0] + b[1] * w.z.to_numpy())))
        r.loc[w.index, "res_wrong"] = w.sel.to_numpy() - p
print("residuals done", flush=True)

rng = np.random.default_rng(42)
quizzes = r[["UserId", "QuizId"]].drop_duplicates()
def split_scores(col, sub):
    """Assign each learner's quizzes to halves at random; return per-learner x group mean residual in each half."""
    q = quizzes.copy(); q["h"] = rng.integers(0, 2, len(q))
    s = sub.merge(q, on=["UserId", "QuizId"])
    a = s.groupby(["UserId", "g", "h"])[col].agg(["mean", "size"]).reset_index()
    a = a[a["size"] >= MINH]
    return a.pivot_table(index="UserId", columns=["g", "h"], values="mean")

def corr_pairs(W):
    """Within r(A_g,B_g) and cross r(A_g,B_h) matrices from one split (pairwise complete)."""
    M = np.full((len(G), len(G)), np.nan); N = np.zeros((len(G), len(G)), int)
    for a, g in enumerate(G):
        for b, h in enumerate(G):
            if (g, 0) not in W or (h, 1) not in W: continue
            x, y = W[(g, 0)], W[(h, 1)]; ok = x.notna() & y.notna()
            if ok.sum() >= 30: M[a, b] = np.corrcoef(x[ok], y[ok])[0, 1]; N[a, b] = ok.sum()
    return M, N

out, mats, draws = {}, {}, {}
for version, col, sub in [("all", "res_all", r), ("wrong", "res_wrong", r[r.res_wrong.notna()])]:
    Ms, Ns, Ws = [], [], []
    for s in range(NS):
        W = split_scores(col, sub); M, N = corr_pairs(W); Ms.append(M); Ns.append(N); Ws.append(W)
    M = np.nanmean(Ms, 0); Msym = (M + M.T) / 2; np.fill_diagonal(Msym, np.diag(M)); mats[version] = Msym
    # bootstrap over learners, pooled over the first 5 splits (split + sampling uncertainty)
    boot = []
    for W in Ws[:5]:
        users = W.index.to_numpy()
        for _ in range(NB // 5):
            Wb = W.loc[rng.choice(users, len(users), replace=True)]; Mb, _ = corr_pairs(Wb); boot.append(Mb)
    boot = np.array(boot)
    rows = []
    for a, g in enumerate(G):
        cross = np.nanmean(np.delete(M[a], a)); within = M[a, a]
        bw = boot[:, a, a]; bc = np.nanmean(np.delete(boot[:, a, :], a, axis=1), 1); bs = bw - bc
        rows.append(dict(group=g, version=version, n_learners=int(np.mean([n[a, a] for n in Ns])), within=within,
                         within_lo=np.nanpercentile(bw, 2.5), within_hi=np.nanpercentile(bw, 97.5),
                         cross=cross, specificity=within - cross,
                         spec_lo=np.nanpercentile(bs, 2.5), spec_hi=np.nanpercentile(bs, 97.5),
                         spec_p=float((bs <= 0).mean())))
    out[version] = pd.DataFrame(rows)
R = pd.concat(out.values()); R["title"] = R.group.map(lambda g: TIT[g][0]); R["verdict"] = R.group.map(AUD)
R.to_csv(RESULTS / "learner_consistency.csv", index=False)
json.dump({v: {"groups": G, "matrix": m.tolist()} for v, m in mats.items()}, open(RESULTS / "learner_consistency_matrix.json", "w"), indent=1)
pd.set_option("display.width", 220)
print(R.drop(columns=["title"]).round(3).to_string(index=False))

# ---------------- figure
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(1, 3, figsize=(13, 4.0), gridspec_kw=dict(width_ratios=[1, 1, 0.95]))
for ax, v, ttl in [(axes[0], "all", "(a) Making the group's error (all attempts)"),
                   (axes[1], "wrong", "(b) Choosing the group's distractor when wrong")]:
    t = out[v].reset_index(drop=True); y = np.arange(len(t))
    ax.barh(y + 0.18, t.within, 0.36, color="#2a78d6", label="same group (within-learner, other quizzes)")
    ax.barh(y - 0.18, t.cross, 0.36, color="#c8c8c8", label="other groups (mean)")
    ax.errorbar(t.within, y + 0.18, xerr=[t.within - t.within_lo, t.within_hi - t.within], fmt="none", ecolor="#1d4f8f", lw=0.8, capsize=2)
    ax.set_yticks(y); ax.set_yticklabels([f"{g}{'' if not AUD[g].startswith('Not') else ' (nc)'}" for g in t.group])
    ax.axvline(0, color="0.5", lw=0.6); ax.set_xlabel("Split-half correlation of ability-adjusted residuals")
    ax.set_title(ttl, fontsize=9.5, loc="left"); ax.grid(axis="x", color="0.92", lw=0.5); ax.set_axisbelow(True)
h, l = axes[0].get_legend_handles_labels(); fig.legend(h, l, loc="lower center", ncol=2, frameon=False, fontsize=8, bbox_to_anchor=(0.36, -0.06))
m = mats["wrong"]; ax = axes[2]
im = ax.imshow(m, cmap="Blues", vmin=0, vmax=max(0.05, np.nanmax(m)))
ax.set_xticks(range(len(G))); ax.set_xticklabels(G); ax.set_yticks(range(len(G))); ax.set_yticklabels(G)
for a in range(len(G)):
    for b in range(len(G)):
        if not np.isnan(m[a, b]): ax.text(b, a, f"{m[a, b]:.2f}", ha="center", va="center", fontsize=6.5, color="white" if m[a, b] > 0.6 * np.nanmax(m) else "0.2")
ax.set_title("(c) Residual correlations when wrong (half 1 × half 2)", fontsize=9.5, loc="left"); ax.spines[:].set_visible(False)
fig.colorbar(im, ax=ax, fraction=0.045)
fig.tight_layout(rect=(0, 0.06, 1, 1)); fig.savefig(FIGURES / "learner_consistency.png", dpi=300, bbox_inches="tight"); print("figure saved")
