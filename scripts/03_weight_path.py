"""
Semantic <-> behavioural weight path (secondary analysis).

Question: does a combined representation form a new joint structure, or does it move between
two distinct structures as the relative weight changes?

Two parameterisations, both on the Protocol-2 training set (629), train-history proficiency,
bootstrap ARI (50 iters, 80% draw + unique, KMeans n_init=20):

 A. Earlier hybrid  h = [PCA30-z(SciBERT) ; beta * z(b)],  k = 6,
    beta in {0 (semantic only), 0.1, 0.25, 0.5, 1, 2, 5, 10, inf (behavioural only)}.
    Also reports the behavioural share of total variance (beta=10 -> 97.1%).

 B. Variance-share path with the BEST semantic representation (tuned, no PCA):
    each block scaled to unit total variance, h = [sqrt(1-w) e ; sqrt(w) b],
    w = behavioural share of total variance in {0, .05, .1, .25, .5, .75, .9, .95, .971, .99, 1}.
    Encoders: Qwen3-Embedding-8B (k=8, primary), MiniLM (k=8), SciBERT (k=6).

At each point: bootstrap ARI, silhouette, ARI vs behavioural-only partition (k=6),
ARI vs that encoder's tuned semantic-only partition, AMI with Eedi level-2 topic, and mean eta^2
of the behavioural features. Output -> results/weight_path.csv
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (EDM, TRAIN_CSV, META, IMAGES, DISTRACTORS, MISC_ZIP, RESULTS, FIGURES, OUTPUTS, ASSETS,
                    SEED as CFG_SEED, GROUP_ENCODER, GROUP_K, emb)
import io, zipfile, ast, warnings; warnings.filterwarnings("ignore")
import os; os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score, adjusted_mutual_info_score
from sklearn.model_selection import train_test_split

SEED, LAMBDA, D = 42, 10.0, 30
df = pd.read_parquet(DISTRACTORS).reset_index(drop=True); N = len(df)
E_sci_cache = emb("SciBERT")
ENC = {"Qwen3-Embedding-8B": (emb("Qwen3-Embedding-8B"), 8), "MiniLM": (emb("all-MiniLM-L6-v2"), 8),
       "SciBERT": (E_sci_cache, 6)}

qids = df.QuestionId.unique()
train_q, _ = train_test_split(qids, test_size=0.30, random_state=SEED)
tr = df.QuestionId.isin(train_q).to_numpy(); assert tr.sum() == 629

# ---- train-history behavioural features (lambda = 10)
QLOCAL = ["n_attempts","p_wrong_option","wrong_rank","option_entropy","difficulty","p_correct","n_wrong_option"]
PROF = ["avg_student_acc_wrong","std_student_acc_wrong","avg_student_acc_correct","discrimination_proxy"]
LOG1P = ["p_wrong_option","p_correct","n_attempts","n_wrong_option"]
raw = pd.read_csv(TRAIN_CSV, usecols=["QuestionId","UserId","AnswerValue","IsCorrect"]).astype(int)
raw_tr = raw[raw.QuestionId.isin(set(train_q.tolist()))]
c_u = raw_tr.groupby("UserId").IsCorrect.sum(); n_u = raw_tr.groupby("UserId").IsCorrect.size(); abar = raw_tr.IsCorrect.mean()
prof = ((c_u + LAMBDA*abar)/(n_u + LAMBDA)).rename("prof").reindex(raw.UserId.unique()).fillna(abar)
r = raw.merge(prof, left_on="UserId", right_index=True)
corr = r[r.IsCorrect==1].groupby("QuestionId").prof.mean().rename("avg_student_acc_correct")
wr = r[r.IsCorrect==0].groupby(["QuestionId","AnswerValue"]).prof.agg(avg_student_acc_wrong="mean", std_student_acc_wrong="std").reset_index()
pf = df[["QuestionId","AnswerValue"]].merge(wr, on=["QuestionId","AnswerValue"], how="left").merge(corr, on="QuestionId", how="left")
pf["discrimination_proxy"] = pf.avg_student_acc_correct - pf.avg_student_acc_wrong
X = df[QLOCAL].copy()
for c in PROF: X[c] = pf[c].values
X = X[QLOCAL+PROF]
for c in X:
    if X[c].isna().any(): X[c] = X[c].fillna(X[c].median())
for c in LOG1P: X[c] = np.log1p(X[c])
E_num = StandardScaler().fit_transform(X.to_numpy()).astype(np.float32)
beh = StandardScaler().fit(E_num[tr]).transform(E_num)[tr].astype(np.float32)   # 629 x 11 (rank column = 0)
FEATS_ETA = [i for i, c in enumerate(QLOCAL+PROF) if c not in ("wrong_rank", "p_correct")]

# ---- topic labels (Eedi level-2) for the training instances
sm = pd.read_csv(META / "subject_metadata.csv", encoding="utf-8-sig")
qm = pd.read_csv(META / "question_metadata_task_3_4.csv", encoding="utf-8-sig")
lvl = dict(zip(sm.SubjectId, sm.Level))
q2t = {q: next((i for i in ast.literal_eval(s) if lvl.get(i) == 2), -1) for q, s in zip(qm.QuestionId, qm.SubjectId)}
topic = df.QuestionId.map(q2t).to_numpy()[tr]

def km(Xm, k): return KMeans(k, random_state=SEED, n_init=20).fit_predict(Xm)
def boot(Xm, k, nb=50):
    ref = km(Xm, k); rng = np.random.default_rng(SEED); n = len(Xm); m = int(n*0.8); a = []
    for _ in range(nb):
        idx = np.unique(rng.choice(n, m, replace=True))
        a.append(adjusted_rand_score(ref[idx], km(Xm[idx], k)))
    return float(np.mean(a)), float(np.std(a))
def eta2(lab):
    out = []
    for j in FEATS_ETA:
        v = beh[:, j]; g = pd.Series(v).groupby(lab)
        out.append(float((g.count()*(g.mean()-v.mean())**2).sum()/((v-v.mean())**2).sum()))
    return float(np.mean(out))
def unit(Z):  # scale a block to unit total variance on the training set
    return Z / np.sqrt(Z.var(axis=0).sum())

LAB_BEH = km(beh, 6)
LAB_BEH_K = {k: km(beh, k) for k in (6, 8)}   # behavioural-only partition at the same k as the path

def evaluate(Xm, k, lab_sem):
    lab = km(Xm, k); m, s = boot(Xm, k)
    return dict(k=k, bootstrap_ari=m, bootstrap_ari_std=s, silhouette=float(silhouette_score(Xm, lab)),
                ari_vs_behavioural=adjusted_rand_score(LAB_BEH, lab), ari_vs_behavioural_same_k=adjusted_rand_score(LAB_BEH_K[k], lab),
                ari_vs_semantic=adjusted_rand_score(lab_sem, lab),
                ami_topic=adjusted_mutual_info_score(topic, lab), eta2_behaviour=eta2(lab))

# ---------------- A. paper parameterisation (PCA30 SciBERT cache, k=6)
pca = PCA(D, random_state=SEED).fit(E_sci_cache[tr])
sem30 = StandardScaler().fit(pca.transform(E_sci_cache[tr])).transform(pca.transform(E_sci_cache[tr])).astype(np.float32)
LAB_SEM30 = km(sem30, 6)
v_sem, v_beh = sem30.var(0).sum(), beh.var(0).sum()
BETAS = [0, 0.1, 0.25, 0.5, 1, 2, 5, 10, np.inf]
def run_beta(b):
    Xm = sem30 if b == 0 else beh if np.isinf(b) else np.hstack([sem30, b*beh])
    share = 0.0 if b == 0 else 1.0 if np.isinf(b) else b*b*v_beh/(v_sem + b*b*v_beh)
    return dict(path="A: earlier hybrid (PCA30 SciBERT)", encoder="SciBERT, PCA30",
                beta=b, behavioural_share=share, **evaluate(Xm, 6, LAB_SEM30))

# ---------------- B. variance-share path with the tuned (no-PCA) semantic representation
W = [0, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.971, 0.99, 1]
SEM = {e: unit(StandardScaler().fit(E[tr]).transform(E[tr]).astype(np.float32)) for e, (E, k) in ENC.items()}
BEHU = unit(beh)
LAB_SEMT = {e: km(SEM[e], ENC[e][1]) for e in ENC}
def run_w(e, w):
    Xm = np.hstack([np.sqrt(1-w)*SEM[e], np.sqrt(w)*BEHU]).astype(np.float32)
    return dict(path="B: variance-share path (tuned semantic, no PCA)", encoder=e, beta=np.nan,
                behavioural_share=w, **evaluate(Xm, ENC[e][1], LAB_SEMT[e]))

jobs = [delayed(run_beta)(b) for b in BETAS] + [delayed(run_w)(e, w) for e in ENC for w in W]
res = pd.DataFrame(Parallel(n_jobs=8, verbose=5)(jobs))
res.to_csv(RESULTS / "weight_path.csv", index=False)
pd.set_option("display.width", 250)
cols = ["encoder","beta","behavioural_share","k","bootstrap_ari","bootstrap_ari_std","silhouette",
        "ari_vs_semantic","ari_vs_behavioural","ami_topic","eta2_behaviour"]
for pth, g in res.groupby("path"):
    print(f"\n=== {pth} ==="); print(g[cols].round(3).to_string(index=False))
