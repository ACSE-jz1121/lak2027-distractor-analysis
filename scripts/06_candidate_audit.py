"""
Candidate misconception groups and their rule-based audit.

  Stage 1  Candidate discovery   config.GROUP_ENCODER (default Qwen3-Embedding-8B), z-scored, no PCA, k-means k = 8.
           MiniLM is used for misconception retrieval, so the external check does not reuse the grouping encoder.
  Stage 2  Candidate audit       for each candidate:
           (a) content coherence  - Eedi topic shares; top-1 misconception matches (MiniLM, 2,587-entry bank);
                                    dominant broad family, all members and 15 centroid-nearest exemplars, each vs a
                                    same-size random-group null (2,000 draws)
           (b) stability          - bootstrap Jaccard of the candidate (100 resamples, full data)
           (c) behavioural evidence - prevalence (selection rate, learners), difficulty, who selects it (selector
                                    proficiency, gap), distractor dominance; 2x2 response profile (easy/hard x
                                    weak-learner / shared-with-proficient) vs the overall distribution;
                                    split-half reliability of within-candidate differences
           (d) held-out check      - Protocol 2: train-fit candidates, 180 held-out distractors assigned with bootstrap
                                    agreement / centroid margin / 95th-pct coverage; train vs test behavioural medians
           (e) rule-based status: content strong = dominant level-2 topic >= 50% and central-15 family p < 0.05;
               moderate = topic >= 40% or all-member family p < 0.05; format-driven if one two-speaker template covers
               >= 60%; unstable if bootstrap Jaccard < 0.60. Supported = strong and not unstable; partly supported =
               moderate and not unstable; otherwise not coherent. Response and held-out evidence do not enter the status.
Outputs -> results/candidate_audit.csv (aggregate table); outputs/audit.json (item-level, contains Eedi item text)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (EDM, TRAIN_CSV, META, IMAGES, DISTRACTORS, MISC_ZIP, RESULTS, FIGURES, OUTPUTS, ASSETS,
                    SEED as CFG_SEED, GROUP_ENCODER, GROUP_K, emb)
import os
import io, zipfile, ast, json, warnings; warnings.filterwarnings("ignore")
from collections import Counter
import numpy as np, pandas as pd
from scipy.stats import mannwhitneyu, spearmanr
from scipy.optimize import linear_sum_assignment
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.model_selection import train_test_split

SEED, K, NEX, NPERM, B_BOOT = 42, 8, 15, 2000, 50
OUT = OUTPUTS; OUT.mkdir(parents=True, exist_ok=True)
df = pd.read_parquet(DISTRACTORS).reset_index(drop=True); N = len(df)
E_qwen = emb(GROUP_ENCODER); E_mini = emb("all-MiniLM-L6-v2")

# ---------------- behavioural evidence features (raw scale)
F = pd.DataFrame({"difficulty": df.difficulty, "selection_rate": df.p_wrong_option,
                  "dominance": (df.p_wrong_option / df.difficulty.clip(lower=1e-6)).clip(upper=1),
                  "selector_prof": df.avg_student_acc_wrong, "correct_prof": df.avg_student_acc_correct,
                  "prof_gap": df.discrimination_proxy, "exposure": df.n_attempts, "learners": df.n_wrong_option})
MED = {f: float(F[f].median()) for f in F}
# 2x2 response profile relative to the whole sample
easy = F.difficulty < MED["difficulty"]; shared = F.prof_gap < MED["prof_gap"]
PROFILE = np.select([easy & ~shared, easy & shared, ~easy & ~shared, ~easy & shared],
                    ["Easy item, weaker learners", "Easy item, shared with proficient learners",
                     "Hard item, weaker learners", "Hard item, shared with proficient learners"], default="")

# ---------------- subject tags
sm = pd.read_csv(META / "subject_metadata.csv", encoding="utf-8-sig")
qm = pd.read_csv(META / "question_metadata_task_3_4.csv", encoding="utf-8-sig")
lvl, nm = dict(zip(sm.SubjectId, sm.Level)), dict(zip(sm.SubjectId, sm.Name))
ids = df.QuestionId.map(qm.set_index("QuestionId").SubjectId.apply(ast.literal_eval))
df["topic"] = ids.apply(lambda s: next((nm[i] for i in s if lvl.get(i) == 2), "(none)"))

# ---------------- misconception bank: MiniLM retrieval (independent of the Qwen3 grouping)
with zipfile.ZipFile(MISC_ZIP) as z: misc = pd.read_csv(io.BytesIO(z.read("misconception_mapping.csv")))
def theme(s):
    s = s.lower()
    if any(w in s for w in ['fraction','numerator','denominator','decimal','standard form','percentage']): return 0
    if any(w in s for w in ['angle','polygon','octagon','pentagon','triangle','square','rectangle','parallelogram','rhombus','diagonal','area','perimeter','vertic','shape','quadrilateral','symmetr','y=','x=','coordinate','reflect','translat','axis','rotat']): return 1
    return 2
FAM = ["Fraction, decimal and representation", "Geometry, spatial and symmetry", "Number, sign and arithmetic"]
from sentence_transformers import SentenceTransformer
M = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2").encode(misc.MisconceptionName.tolist(), normalize_embeddings=True,
                                                                          batch_size=128, show_progress_bar=False).astype(np.float32)
S = E_mini @ M.T; top1 = S.argmax(1); top1_sim = S.max(1)
fam = np.array([theme(misc.MisconceptionName.iloc[t]) for t in top1])
mname = misc.MisconceptionName.to_numpy()

# ---------------- Stage 1: candidates (Protocol 1, full data)
Xq = StandardScaler().fit_transform(E_qwen).astype(np.float32)
km = KMeans(K, random_state=SEED, n_init=20).fit(Xq); lab, cen = km.labels_, km.cluster_centers_
NAMES = {}  # assigned below from evidence (topic + exemplars), checked by hand
def is_template(idx):
    t = df.text_input.iloc[idx]
    return float(t.str.contains(r"Tom and Katie|Jo and Paul|Tom says|Katie says").mean())

# per-candidate stability: bootstrap Jaccard
rng = np.random.default_rng(SEED); jac = {c: [] for c in range(K)}
for _ in range(100):
    idx = rng.choice(N, N, replace=True); u = np.unique(idx)
    lb = KMeans(K, random_state=SEED, n_init=10).fit(Xq[idx]).predict(Xq[u])
    for c in range(K):
        A = set(u[lab[u] == c])
        if A: jac[c].append(max(len(A & set(u[lb == j])) / len(A | set(u[lb == j])) for j in range(K)))

# null distributions for dominant-family share in random groups of a given size
null_cache = {}
def null_share(n):
    if n not in null_cache:
        r = np.random.default_rng(SEED + n)
        null_cache[n] = np.array([np.bincount(fam[r.choice(N, n, replace=False)], minlength=3).max() / n for _ in range(NPERM)])
    return null_cache[n]

# split-half reliability of the behavioural features, used to judge within-candidate differences
raw = pd.read_csv(TRAIN_CSV, usecols=["QuestionId","UserId","AnswerValue","IsCorrect"]).astype(int)
acc = raw.groupby("UserId").IsCorrect.mean().rename("acc"); raw = raw.merge(acc, left_on="UserId", right_index=True)
half = set(np.random.default_rng(SEED).choice(raw.UserId.unique(), raw.UserId.nunique() // 2, replace=False))
def half_feats(sub):
    q = sub.groupby("QuestionId").agg(n=("UserId", "size"), pc=("IsCorrect", "mean"))
    cp = sub[sub.IsCorrect == 1].groupby("QuestionId").acc.mean().rename("cp")
    ch = sub.groupby(["QuestionId", "AnswerValue"]).agg(nc=("UserId", "size"), sp=("acc", "mean")).reset_index()
    o = df[["QuestionId", "AnswerValue"]].merge(ch, on=["QuestionId", "AnswerValue"], how="left").merge(q, on="QuestionId", how="left").merge(cp, on="QuestionId", how="left")
    return pd.DataFrame({"selection_rate": o.nc / o.n, "selector_prof": o.sp, "prof_gap": o.cp - o.sp, "difficulty": 1 - o.pc})
HA, HB = half_feats(raw[raw.UserId.isin(half)]), half_feats(raw[~raw.UserId.isin(half)])

def auc(a, b): return float(mannwhitneyu(a, b).statistic / (len(a) * len(b)))
def q3(v): return [float(np.percentile(v, 25)), float(np.median(v)), float(np.percentile(v, 75))]
def ex_row(i, why=None):
    r = df.iloc[i]
    d = dict(question=" ".join(str(r.llm_question).split())[:220], correct=str(r.correct_text)[:60], chosen=str(r.distractor_text)[:60],
             topic=r.topic, misconception=str(mname[top1[i]]), match_sim=round(float(top1_sim[i]), 3),
             family=FAM[fam[i]], profile=str(PROFILE[i]),
             **{f: round(float(F.loc[i, f]), 3) for f in ["difficulty", "selection_rate", "dominance", "selector_prof", "prof_gap"]},
             exposure=int(F.loc[i, "exposure"]), learners=int(F.loc[i, "learners"]))
    if why: d["why"] = why
    return d

cands = []
for c in range(K):
    idx = np.where(lab == c)[0]; n = len(idx)
    order = idx[np.argsort(((Xq[idx] - cen[c]) ** 2).sum(1))]; ex = order[:NEX]
    tp = df.topic.iloc[idx].value_counts(normalize=True)
    cnt_all = np.bincount(fam[idx], minlength=3); cnt_ex = np.bincount(fam[ex], minlength=3)
    share_all, share_ex = cnt_all.max() / n, cnt_ex.max() / NEX
    p_all = float((null_share(n) >= share_all).mean()); p_ex = float((null_share(NEX) >= share_ex).mean())
    mc = Counter(mname[top1[idx]]).most_common(5)
    # behaviour
    beh = {f: q3(F.loc[idx, f]) for f in ["difficulty", "selection_rate", "dominance", "selector_prof", "prof_gap", "exposure"]}
    vs_rest = {f: auc(F.loc[idx, f], F.loc[np.setdiff1d(np.arange(N), idx), f]) for f in ["difficulty", "selection_rate", "selector_prof", "prof_gap", "dominance"]}
    prof = pd.Series(PROFILE[idx]).value_counts(normalize=True).to_dict()
    rel = {}
    for f in ["selection_rate", "selector_prof", "prof_gap"]:
        a, b = HA.loc[idx, f], HB.loc[idx, f]; ok = a.notna() & b.notna()
        rel[f] = float(spearmanr(a[ok], b[ok]).correlation)
    # within-candidate contrasting distractors
    sub = F.loc[idx]
    contrasts = [ex_row(int(sub.prof_gap.idxmin()), "Smallest proficiency gap: also chosen by relatively proficient learners"),
                 ex_row(int(sub.prof_gap.idxmax()), "Largest proficiency gap: chosen mainly by weaker learners"),
                 ex_row(int(sub.selection_rate.idxmax()), "Most frequently selected distractor"),
                 ex_row(int(sub.selection_rate.idxmin()), "Least frequently selected distractor")]
    cands.append(dict(id=f"S{c}", n=n, template_share=is_template(idx),
                      topics=[dict(topic=t, share=float(s)) for t, s in tp.head(4).items()],
                      family=dict(dominant=FAM[cnt_all.argmax()], share_all=float(share_all), null_all=float(null_share(n).mean()), p_all=p_all,
                                  dominant_ex=FAM[cnt_ex.argmax()], share_ex=float(share_ex), null_ex=float(null_share(NEX).mean()), p_ex=p_ex,
                                  counts=[int(x) for x in cnt_all]),
                      misconceptions=[dict(name=str(m), count=int(k)) for m, k in mc], match_sim_median=float(np.median(top1_sim[idx])),
                      stability=dict(jaccard=float(np.mean(jac[c])), jaccard_sd=float(np.std(jac[c]))),
                      behaviour=beh, auc_vs_rest=vs_rest, profile_shares=prof,
                      learners_total=int(F.loc[idx, "learners"].sum()), within_reliability=rel,
                      exemplars=[ex_row(int(i)) for i in ex[:6]], contrasts=contrasts))

# ---------------- held-out check (Protocol 2): train-fit candidates, assign 180 test distractors
qids = df.QuestionId.unique()
tq, tmp = train_test_split(qids, test_size=0.30, random_state=SEED); vq, teq = train_test_split(tmp, test_size=2/3, random_state=SEED)
tr = df.QuestionId.isin(tq).to_numpy(); te = df.QuestionId.isin(teq).to_numpy(); tr_idx = np.where(tr)[0]
def rep(fit_mask): return StandardScaler().fit(E_qwen[fit_mask]).transform(E_qwen).astype(np.float32)
Xp = rep(tr); kp = KMeans(K, random_state=SEED, n_init=20).fit(Xp[tr]); ref_tr, cen_tr = kp.labels_, kp.cluster_centers_
# map train-fit clusters to the full-data candidate ids (Hungarian on the training instances)
cm = pd.crosstab(pd.Series(lab[tr]), pd.Series(ref_tr)).reindex(index=range(K), columns=range(K), fill_value=0).to_numpy()
rr, cc = linear_sum_assignment(-cm); to_full = {int(cc[i]): int(rr[i]) for i in range(K)}
d_te = np.linalg.norm(Xp[te][:, None] - cen_tr[None], axis=2); o = np.argsort(d_te, 1); a_te = o[:, 0]
d1 = d_te[np.arange(te.sum()), o[:, 0]]; d2 = d_te[np.arange(te.sum()), o[:, 1]]; margin = (d2 - d1) / d2
tau = np.array([np.quantile(np.linalg.norm(Xp[tr][ref_tr == k] - cen_tr[k], axis=1), 0.95) for k in range(K)]); covered = d1 <= tau[a_te]
rngb = np.random.default_rng(SEED); boot = np.zeros((te.sum(), B_BOOT), int)
for b in range(B_BOOT):
    sel = np.unique(rngb.choice(tr_idx, len(tr_idx), replace=True)); fm = np.zeros(N, bool); fm[sel] = True
    Xb = rep(fm); kb = KMeans(K, random_state=SEED, n_init=10).fit(Xb[sel])
    cmb = pd.crosstab(pd.Series(ref_tr), pd.Series(kb.predict(Xb[tr]))).reindex(index=range(K), columns=range(K), fill_value=0).to_numpy()
    r2, c2 = linear_sum_assignment(-cmb); mp = {int(c2[i]): int(r2[i]) for i in range(K)}
    boot[:, b] = [mp[p] for p in kb.predict(Xb[te])]
agree = np.array([np.bincount(row).max() / B_BOOT for row in boot])
te_idx = np.where(te)[0]; a_full = np.array([to_full[int(a)] for a in a_te])
heldout = dict(n_test=int(te.sum()), mean_agreement=float(agree.mean()), prop_agree_08=float((agree >= 0.8).mean()),
               median_margin=float(np.median(margin)), coverage_95=float(covered.mean()),
               full_vs_trainfit_ari_on_train=float(adjusted_rand_score(lab[tr], ref_tr)))
for cnd in cands:
    c = int(cnd["id"][1:]); m = a_full == c; ti = te_idx[m]; trn = np.where(tr & (lab == c))[0]
    cnd["heldout"] = dict(n_assigned=int(m.sum()), mean_agreement=float(agree[m].mean()) if m.any() else None,
                          coverage=float(covered[m].mean()) if m.any() else None,
                          topic_match=float((df.topic.iloc[ti].to_numpy() == cnd["topics"][0]["topic"]).mean()) if m.any() else None,
                          family_match=float((fam[ti] == FAM.index(cnd["family"]["dominant"])).mean()) if m.any() else None,
                          train_median_selection=float(F.selection_rate.iloc[trn].median()),
                          test_median_selection=float(F.selection_rate.iloc[ti].median()) if m.any() else None,
                          train_median_selector_prof=float(F.selector_prof.iloc[trn].median()),
                          test_median_selector_prof=float(F.selector_prof.iloc[ti].median()) if m.any() else None)

# ---------------- evidence-rule verdicts (transparent thresholds)
for cnd in cands:
    top = cnd["topics"][0]["share"]; fx = cnd["family"]; st = cnd["stability"]["jaccard"]
    if cnd["template_share"] >= 0.6: content = "format-driven"
    elif top >= 0.5 and fx["p_ex"] < 0.05: content = "strong"   # majority topic
    elif top >= 0.4 or fx["p_all"] < 0.05: content = "moderate"
    else: content = "weak"
    stab = "stable" if st >= 0.75 else "moderate" if st >= 0.6 else "unstable"
    dominant_profile = max(cnd["profile_shares"].values())
    behaviour = "consistent" if dominant_profile >= 0.45 else "heterogeneous"
    if content == "strong" and stab != "unstable": verdict = "Supported candidate"
    elif content in ("strong", "moderate") and stab != "unstable": verdict = "Partly supported"
    else: verdict = "Not a coherent misconception candidate"
    cnd["audit"] = dict(content=content, stability=stab, behaviour=behaviour, dominant_profile_share=dominant_profile, verdict=verdict)

context = dict(overall_medians=MED)
audit = dict(candidates=cands, heldout=heldout, context=context,
             meta=dict(n=N, k=K, grouping_encoder=GROUP_ENCODER, retrieval_encoder="all-MiniLM-L6-v2", bank_size=int(len(misc))))
json.dump(audit, open(OUT / "audit.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
pd.DataFrame([dict(id=c["id"], n=c["n"], top_topic=c["topics"][0]["topic"], top_share=c["topics"][0]["share"], template=c["template_share"],
                   family=c["family"]["dominant"], fam_all=c["family"]["share_all"], p_all=c["family"]["p_all"], fam_ex=c["family"]["share_ex"], p_ex=c["family"]["p_ex"],
                   jaccard=c["stability"]["jaccard"], sel_med=c["behaviour"]["selection_rate"][1], selprof_med=c["behaviour"]["selector_prof"][1],
                   gap_med=c["behaviour"]["prof_gap"][1], diff_med=c["behaviour"]["difficulty"][1], dom_profile=c["audit"]["dominant_profile_share"],
                   heldout_n=c["heldout"]["n_assigned"], heldout_agree=c["heldout"]["mean_agreement"], verdict=c["audit"]["verdict"],
                   content=c["audit"]["content"], top_misconception=c["misconceptions"][0]["name"]) for c in cands]).to_csv(RESULTS / "candidate_audit.csv", index=False)
pd.set_option("display.width", 260); pd.set_option("display.max_colwidth", 70)
print(pd.read_csv(RESULTS / "candidate_audit.csv").round(3).to_string(index=False)); print(json.dumps(heldout, indent=1))
for c in cands:
    print(f"\n{c['id']} n={c['n']} {c['topics'][:3]}\n  misconceptions: {[m['name'][:60]+' ('+str(m['count'])+')' for m in c['misconceptions'][:4]]}")
    print("  profiles:", {k[:30]: round(v, 2) for k, v in c["profile_shares"].items()}, "| reliability", {k: round(v, 2) for k, v in c["within_reliability"].items()})
