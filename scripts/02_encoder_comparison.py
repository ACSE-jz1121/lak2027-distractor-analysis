"""
Semantic-encoder comparison (RQ1): every encoder cached by 01_encode.py, evaluated with one protocol.

Per encoder (results cached per encoder in cache/encoder_eval/<name>.json):
  tuning   : 629 Protocol-2 training instances, k in {4,6,8,10,12,15} x d in {no PCA,30,50,70,100,150,200},
             bootstrap ARI (50 x 80% + unique, KMeans n_init=20), selection by ARI -> silhouette -> DB
  at the selected setting AND at a fixed k=8 (no PCA) for like-for-like comparison, on the full 899 (Protocol 1):
             NMI with the most specific Eedi subject tag, AMI with level-2 topic,
             ARI with the behavioural-only partition (same k), mean eta^2 of 9 behavioural features, eta^2 of text length,
             external misconception alignment (all instances and 15 centroid-nearest exemplars, 2,000 permutations),
             with two retrievers (MiniLM and Qwen3-0.6B) so that no encoder is only checked by itself
  held-out : Protocol 2 at the selected setting: 180 test distractors, bootstrap assignment agreement (B=50),
             centroid margin, 95th-percentile coverage
Outputs -> results/encoder_comparison.csv, results/encoder_tuning_grid.csv, results/encoder_pairwise_ari_k8.csv
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (EDM, TRAIN_CSV, META, IMAGES, DISTRACTORS, MISC_ZIP, RESULTS, FIGURES, OUTPUTS, ASSETS,
                    SEED as CFG_SEED, GROUP_ENCODER, GROUP_K, emb)
import io, sys, json, zipfile, ast, time, warnings; warnings.filterwarnings("ignore")
import os; os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from scipy.optimize import linear_sum_assignment
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score, adjusted_mutual_info_score,
                             silhouette_score, calinski_harabasz_score, davies_bouldin_score)
from sklearn.model_selection import train_test_split

SEED, NBOOT, NPERM, NEX = 42, 50, 2000, 15
K_GRID, D_GRID = [4, 6, 8, 10, 12, 15], [None, 30, 50, 70, 100, 150, 200]
EVAL = Path(__file__).resolve().parents[1] / "cache" / "encoder_eval"; EVAL.mkdir(parents=True, exist_ok=True)
df = pd.read_parquet(DISTRACTORS).reset_index(drop=True); N = len(df)

# ---------------- encoders
ENCODERS = ["SciBERT", "all-MiniLM-L6-v2", "all-mpnet-base-v2", "bge-large-en-v1.5", "e5-large-v2", "mxbai-embed-large-v1",
            "nomic-embed-text-v1.5", "Qwen3-Embedding-0.6B", "Qwen3-Embedding-4B", "Qwen3-Embedding-8B"]
ENC = {e: emb(e) for e in ENCODERS}
print("encoders:", list(ENC), flush=True)

# ---------------- split, behavioural reference partitions, labels
qids = df.QuestionId.unique()
tq, tmp = train_test_split(qids, test_size=0.30, random_state=SEED); vq, teq = train_test_split(tmp, test_size=2/3, random_state=SEED)
tr = df.QuestionId.isin(tq).to_numpy(); te = df.QuestionId.isin(teq).to_numpy(); tr_idx = np.where(tr)[0]
assert tr.sum() == 629 and te.sum() == 180
# train-history behavioural features, (lambda = 10), shared with 03_weight_path.py and 05_topic_baseline.py
QLOCAL = ["n_attempts","p_wrong_option","wrong_rank","option_entropy","difficulty","p_correct","n_wrong_option"]
PROF = ["avg_student_acc_wrong","std_student_acc_wrong","avg_student_acc_correct","discrimination_proxy"]
raw = pd.read_csv(TRAIN_CSV, usecols=["QuestionId","UserId","AnswerValue","IsCorrect"]).astype(int)
raw_tr = raw[raw.QuestionId.isin(set(tq.tolist()))]
c_u = raw_tr.groupby("UserId").IsCorrect.sum(); n_u = raw_tr.groupby("UserId").IsCorrect.size(); abar = raw_tr.IsCorrect.mean()
prof = ((c_u + 10*abar)/(n_u + 10)).rename("prof").reindex(raw.UserId.unique()).fillna(abar)
r_ = raw.merge(prof, left_on="UserId", right_index=True)
corr = r_[r_.IsCorrect==1].groupby("QuestionId").prof.mean().rename("avg_student_acc_correct")
wr = r_[r_.IsCorrect==0].groupby(["QuestionId","AnswerValue"]).prof.agg(avg_student_acc_wrong="mean", std_student_acc_wrong="std").reset_index()
pf = df[["QuestionId","AnswerValue"]].merge(wr, on=["QuestionId","AnswerValue"], how="left").merge(corr, on="QuestionId", how="left")
pf["discrimination_proxy"] = pf.avg_student_acc_correct - pf.avg_student_acc_wrong
Xb = df[QLOCAL].copy()
for c in PROF: Xb[c] = pf[c].values
for c in Xb:
    if Xb[c].isna().any(): Xb[c] = Xb[c].fillna(Xb[c].median())
for c in ["p_wrong_option","p_correct","n_attempts","n_wrong_option"]: Xb[c] = np.log1p(Xb[c])
beh = StandardScaler().fit_transform(Xb.to_numpy())
BEH = {k: KMeans(k, random_state=SEED, n_init=20).fit_predict(beh) for k in K_GRID}
Zf = pd.DataFrame(beh, columns=list(Xb.columns))
FEATS = ["difficulty","p_wrong_option","discrimination_proxy","avg_student_acc_wrong","std_student_acc_wrong",
         "avg_student_acc_correct","option_entropy","n_attempts","n_wrong_option"]
loglen = np.log(df.text_input.str.len().to_numpy().astype(float))
sm = pd.read_csv(META / "subject_metadata.csv", encoding="utf-8-sig")
qm = pd.read_csv(META / "question_metadata_task_3_4.csv", encoding="utf-8-sig")
lvl = dict(zip(sm.SubjectId, sm.Level))
ids = df.QuestionId.map(qm.set_index("QuestionId").SubjectId.apply(ast.literal_eval))
leaf = ids.apply(lambda s: max(s, key=lambda i: lvl.get(i, 0))).astype(str).to_numpy()
topic = ids.apply(lambda s: next((i for i in s if lvl.get(i) == 2), -1)).astype(str).to_numpy()

# ---------------- misconception families with two retrievers
with zipfile.ZipFile(MISC_ZIP) as z:
    misc = pd.read_csv(io.BytesIO(z.read("misconception_mapping.csv")))
def theme(s):
    s = s.lower()
    if any(w in s for w in ['fraction','numerator','denominator','decimal','standard form','percentage']): return 0
    if any(w in s for w in ['angle','polygon','octagon','pentagon','triangle','square','rectangle','parallelogram','rhombus','diagonal','area','perimeter','vertic','shape','quadrilateral','symmetr','y=','x=','coordinate','reflect','translat','axis','rotat']): return 1
    return 2
bank_fam = np.array([theme(s) for s in misc.MisconceptionName])
Mm = emb("bank_minilm")
Mq = emb("bank_qwen3-0.6b")
FAMS = {"MiniLM": bank_fam[(ENC["all-MiniLM-L6-v2"] @ Mm.T).argmax(1)], "Qwen3-0.6B": bank_fam[(ENC["Qwen3-Embedding-0.6B"] @ Mq.T).argmax(1)]}

# ---------------- helpers
def km(X, k, n_init=20): return KMeans(k, random_state=SEED, n_init=n_init).fit(X)
def rep(E, d, fit):
    if d is None: return StandardScaler().fit(E[fit]).transform(E).astype(np.float32)
    p = PCA(min(d, E.shape[1], int(fit.sum())), random_state=SEED).fit(E[fit]); Z = p.transform(E)
    return StandardScaler().fit(Z[fit]).transform(Z).astype(np.float32)
def boot(Xtr, k):
    ref = km(Xtr, k).labels_; r = np.random.default_rng(SEED); n = len(Xtr); m = int(n * 0.8); a = []
    for _ in range(NBOOT):
        i = np.unique(r.choice(n, m, replace=True)); a.append(adjusted_rand_score(ref[i], km(Xtr[i], k).labels_))
    return float(np.mean(a)), float(np.std(a))
def tune_one(name, d, k):
    X = rep(ENC[name], d, tr)[tr]; lab = km(X, k).labels_; m, s = boot(X, k)
    return dict(encoder=name, k=k, d="no_pca" if d is None else d, Mean_ARI=m, Std=s, Sil=float(silhouette_score(X, lab)),
                CH=float(calinski_harabasz_score(X, lab)), DB=float(davies_bouldin_score(X, lab)))
def eta2(lab, v):
    g = pd.Series(v).groupby(lab); return float((g.count() * (g.mean() - v.mean()) ** 2).sum() / ((v - v.mean()) ** 2).sum())
def align(lab, X, cen, fam):
    def cov_all(l): return float(np.mean([np.bincount(fam[l == c], minlength=3).max() / (l == c).sum() for c in np.unique(l)]))
    def cov_ex(l, cn=None):
        o = []
        for c in np.unique(l):
            i = np.where(l == c)[0]; ct = cn[c] if cn is not None else X[i].mean(0)
            e = i[np.argsort(((X[i] - ct) ** 2).sum(1))[:NEX]]; o.append(np.bincount(fam[e], minlength=3).max() / len(e))
        return float(np.mean(o))
    r = np.random.default_rng(SEED); a = cov_all(lab); na = np.array([cov_all(r.permutation(lab)) for _ in range(NPERM)])
    e = cov_ex(lab, cen); r = np.random.default_rng(SEED); ne = np.array([cov_ex(r.permutation(lab)) for _ in range(NPERM)])
    return dict(all=a, all_null=float(na.mean()), all_p=float((na >= a).mean()), ex=e, ex_null=float(ne.mean()), ex_p=float((ne >= e).mean()))
def full_eval(name, k, d):
    X = rep(ENC[name], d, np.ones(N, bool)); f = km(X, k); lab, cen = f.labels_, f.cluster_centers_
    out = dict(NMI_leaf=normalized_mutual_info_score(leaf, lab), AMI_topic=adjusted_mutual_info_score(topic, lab),
               ARI_vs_behavioural=adjusted_rand_score(BEH[k], lab), eta2_behaviour=float(np.mean([eta2(lab, Zf[c].to_numpy()) for c in FEATS])),
               eta2_text_length=eta2(lab, loglen), sizes=sorted(np.bincount(lab).tolist(), reverse=True))
    for rn, fam in FAMS.items():
        for kk, v in align(lab, X, cen, fam).items(): out[f"align_{rn}_{kk}"] = v
    return out, lab
def hmap(ref, pred, K):
    cm = pd.crosstab(pd.Series(ref), pd.Series(pred)).reindex(index=range(K), columns=range(K), fill_value=0).to_numpy()
    r, c = linear_sum_assignment(-cm); return {int(c[i]): int(r[i]) for i in range(K)}
def heldout(name, k, d):
    E = ENC[name]; X = rep(E, d, tr); f = km(X[tr], k); ref, cen = f.labels_, f.cluster_centers_
    dt = np.linalg.norm(X[te][:, None] - cen[None], axis=2); o = np.argsort(dt, 1); a = o[:, 0]
    d1 = dt[np.arange(te.sum()), o[:, 0]]; d2 = dt[np.arange(te.sum()), o[:, 1]]
    tau = np.array([np.quantile(np.linalg.norm(X[tr][ref == c] - cen[c], axis=1), 0.95) for c in range(k)])
    r = np.random.default_rng(SEED); bt = np.zeros((te.sum(), NBOOT), int)
    for b in range(NBOOT):
        sel = np.unique(r.choice(tr_idx, len(tr_idx), replace=True)); fm = np.zeros(N, bool); fm[sel] = True
        Xb = rep(E, d, fm); kb = km(Xb[sel], k, 10); mp = hmap(ref, kb.predict(Xb[tr]), k)
        bt[:, b] = [mp[p] for p in kb.predict(Xb[te])]
    ag = np.array([np.bincount(row).max() / NBOOT for row in bt])
    return dict(heldout_agreement=float(ag.mean()), heldout_agree08=float((ag >= 0.8).mean()),
                heldout_margin=float(np.median((d2 - d1) / d2)), heldout_coverage95=float((d1 <= tau[a]).mean()))

# ---------------- run (per-encoder cache)
labels = {}
for name in ENC:
    cp = EVAL / f"{name}.json"
    if cp.exists():
        res = json.load(open(cp)); labels[name] = np.array(res.pop("_labels_k8")); print(f"[cached] {name}", flush=True); continue
    t0 = time.time(); print(f"[eval] {name} dim={ENC[name].shape[1]}", flush=True)
    grid = pd.DataFrame(Parallel(n_jobs=6)(delayed(tune_one)(name, d, k) for d in D_GRID for k in K_GRID))
    grid["d"] = grid["d"].astype(str)
    grid = grid.sort_values(["Mean_ARI", "Sil", "DB"], ascending=[False, False, True]).reset_index(drop=True)
    b = grid.iloc[0]; bd = None if b.d == "no_pca" else int(float(b.d)); bk = int(b.k)
    f8 = grid[(grid.k == 8) & (grid.d == "no_pca")].iloc[0]
    sel_eval, _ = full_eval(name, bk, bd); k8_eval, lab8 = full_eval(name, 8, None)
    res = dict(encoder=name, dim=int(ENC[name].shape[1]), selected_k=bk, selected_d=b.d, stability=float(b.Mean_ARI), stability_sd=float(b.Std),
               silhouette=float(b.Sil), k8_stability=float(f8.Mean_ARI), k8_stability_sd=float(f8.Std),
               **{f"sel_{k}": v for k, v in sel_eval.items()}, **{f"k8_{k}": v for k, v in k8_eval.items()},
               **heldout(name, bk, bd), grid=grid.to_dict("records"), seconds=round(time.time() - t0))
    json.dump({**res, "_labels_k8": lab8.tolist()}, open(cp, "w"), default=float); labels[name] = lab8
    print(f"   done in {res['seconds']}s: k={bk} d={b.d} stability={b.Mean_ARI:.3f} | k8 stability={f8.Mean_ARI:.3f} "
          f"| k8 align(Qwen retr.)={k8_eval['align_Qwen3-0.6B_all']:.3f} | k8 ARI vs beh={k8_eval['ARI_vs_behavioural']:.3f}", flush=True)

# ---------------- summaries
rows, grids = [], []
for name in ENC:
    r = json.load(open(EVAL / f"{name}.json")); r.pop("_labels_k8"); g = pd.DataFrame(r.pop("grid")); g["encoder"] = name; grids.append(g)
    r["sel_sizes"] = str(r["sel_sizes"]); r["k8_sizes"] = str(r["k8_sizes"]); rows.append(r)
S = pd.DataFrame(rows); S.to_csv(RESULTS / "encoder_comparison.csv", index=False)
pd.concat(grids).to_csv(RESULTS / "encoder_tuning_grid.csv", index=False)
names = list(labels); P = pd.DataFrame([[adjusted_rand_score(labels[a], labels[b]) for b in names] for a in names], index=names, columns=names)
P.to_csv(RESULTS / "encoder_pairwise_ari_k8.csv")
pd.set_option("display.width", 250)
print(S[["encoder", "dim", "selected_k", "selected_d", "stability", "k8_stability", "k8_NMI_leaf", "k8_ARI_vs_behavioural",
         "k8_align_Qwen3-0.6B_all", "k8_align_Qwen3-0.6B_ex", "k8_align_MiniLM_all", "k8_eta2_text_length", "heldout_agreement", "heldout_coverage95"]].round(3).to_string(index=False))
