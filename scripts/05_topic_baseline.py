"""Is semantic-cluster alignment with the misconception bank more than topic alignment?

(1) Topic baselines: the same coverage measure computed for groupings defined by Eedi subject tags
    (level 1, level 2, most specific tag, and the 7 largest level-2 topics + other = 8 groups),
    and semantic k-means at the SAME number of groups as each tag level.
(2) Within-topic test: cluster labels are permuted only within each topic (level-2 tag, and most specific tag),
    so every permuted partition has exactly the same topic composition per cluster. If observed coverage exceeds
    this null, clusters capture misconception-family structure that topic alone does not explain.
(3) Misconception-level (not family) agreement: AMI between partition and the retrieved misconception id,
    also with the within-topic null.
(4) Retrieval-free check on the subset of distractors linked to human-annotated Eedi 2024 items
    (exp_match_2024_labels.py, similarity >= 0.80): retrieval accuracy against the annotation, and pairwise
    same-annotated-misconception rates for pairs sharing a semantic cluster vs a topic tag.
Retrievers: MiniLM (paper) and Qwen3-Embedding-0.6B.  Outputs -> results/topic_baseline/
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (EDM, TRAIN_CSV, META, IMAGES, DISTRACTORS, MISC_ZIP, RESULTS, FIGURES, OUTPUTS, ASSETS,
                    SEED as CFG_SEED, GROUP_ENCODER, GROUP_K, emb)
import io, ast, json, zipfile, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.model_selection import train_test_split

SEED, NPERM = 42, 2000
EVAL = Path(__file__).resolve().parents[1] / "cache" / "encoder_eval"   # written by 02_encoder_comparison.py
df = pd.read_parquet(DISTRACTORS).reset_index(drop=True); N = len(df)

# ---------------- topic labels
sm = pd.read_csv(META / "subject_metadata.csv", encoding="utf-8-sig")
qm = pd.read_csv(META / "question_metadata_task_3_4.csv", encoding="utf-8-sig")
lvl = dict(zip(sm.SubjectId, sm.Level)); name = dict(zip(sm.SubjectId, sm.Name))
ids = df.QuestionId.map(qm.set_index("QuestionId").SubjectId.apply(ast.literal_eval))
def at(level): return ids.apply(lambda s: next((i for i in s if lvl.get(i) == level), -1)).to_numpy()
L1, L2 = at(1), at(2)
LEAF = ids.apply(lambda s: max(s, key=lambda i: lvl.get(i, 0))).to_numpy()
top7 = pd.Series(L2).value_counts().index[:7]
L2_8 = np.where(np.isin(L2, top7), L2, -99)
def codes(x): return pd.factorize(x)[0]
TOPIC = {"Eedi level-1 subject": codes(L1), "Eedi level-2 topic, 7 largest + other": codes(L2_8),
         "Eedi level-2 topic": codes(L2), "Eedi most specific tag": codes(LEAF)}

# ---------------- misconception bank, retrievers, families
with zipfile.ZipFile(MISC_ZIP) as z:
    misc = pd.read_csv(io.BytesIO(z.read("misconception_mapping.csv"))); tr24 = pd.read_csv(io.BytesIO(z.read("train.csv")))
def theme(s):  # fixed keyword rule for the three misconception families
    s = s.lower()
    if any(w in s for w in ['fraction','numerator','denominator','decimal','standard form','percentage']): return 0
    if any(w in s for w in ['angle','polygon','octagon','pentagon','triangle','square','rectangle','parallelogram','rhombus','diagonal','area','perimeter','vertic','shape','quadrilateral','symmetr','y=','x=','coordinate','reflect','translat','axis','rotat']): return 1
    return 2
bank_fam = np.array([theme(s) for s in misc.MisconceptionName])
E_mini = emb("all-MiniLM-L6-v2"); E_qwen = emb("Qwen3-Embedding-0.6B")
M_mini = emb("bank_minilm")
M_qwen = emb("bank_qwen3-0.6b")
SIMS = {"MiniLM": E_mini @ M_mini.T, "Qwen3-0.6B": E_qwen @ M_qwen.T}
TOP1 = {r: s.argmax(1) for r, s in SIMS.items()}
FAM = {r: bank_fam[t] for r, t in TOP1.items()}

# ---------------- partitions
PART = dict(TOPIC)
ENC_FILES = {"SciBERT": "SciBERT", "MiniLM": "all-MiniLM-L6-v2", "Qwen3-0.6B": "Qwen3-Embedding-0.6B",
             "mpnet-base": "all-mpnet-base-v2", "bge-large": "bge-large-en-v1.5", "e5-large": "e5-large-v2",
             "mxbai-large": "mxbai-embed-large-v1", "nomic-v1.5": "nomic-embed-text-v1.5", "Qwen3-4B": "Qwen3-Embedding-4B",
             "Qwen3-8B": "Qwen3-Embedding-8B"}
for short, f in ENC_FILES.items():
    p = EVAL / f"{f}.json"
    if p.exists(): PART[f"Semantic {short} (k=8)"] = np.array(json.load(open(p))["_labels_k8"])
EMB = {"MiniLM": E_mini, "Qwen3-0.6B": E_qwen}
EMB["Qwen3-8B"] = emb("Qwen3-Embedding-8B")
for short, E in EMB.items():
    X = StandardScaler().fit_transform(E)
    for tname in ["Eedi level-2 topic", "Eedi most specific tag"]:
        k = len(np.unique(TOPIC[tname]))
        PART[f"Semantic {short} (k={k}, = {tname.split(' ', 1)[1]})"] = KMeans(k, random_state=SEED, n_init=20).fit_predict(X)
# behavioural (train-history features, as in 02_encoder_comparison.py)
qids = df.QuestionId.unique(); tq, _ = train_test_split(qids, test_size=0.30, random_state=SEED)
QLOCAL = ["n_attempts","p_wrong_option","wrong_rank","option_entropy","difficulty","p_correct","n_wrong_option"]
raw = pd.read_csv(TRAIN_CSV, usecols=["QuestionId","UserId","AnswerValue","IsCorrect"]).astype(int)
rt = raw[raw.QuestionId.isin(set(tq.tolist()))]; abar = rt.IsCorrect.mean()
prof = ((rt.groupby("UserId").IsCorrect.sum() + 10*abar) / (rt.groupby("UserId").IsCorrect.size() + 10)).rename("prof").reindex(raw.UserId.unique()).fillna(abar)
r_ = raw.merge(prof, left_on="UserId", right_index=True)
corr = r_[r_.IsCorrect==1].groupby("QuestionId").prof.mean().rename("avg_student_acc_correct")
wr = r_[r_.IsCorrect==0].groupby(["QuestionId","AnswerValue"]).prof.agg(avg_student_acc_wrong="mean", std_student_acc_wrong="std").reset_index()
pf = df[["QuestionId","AnswerValue"]].merge(wr, on=["QuestionId","AnswerValue"], how="left").merge(corr, on="QuestionId", how="left")
pf["discrimination_proxy"] = pf.avg_student_acc_correct - pf.avg_student_acc_wrong
Xb = df[QLOCAL].copy()
for c in ["avg_student_acc_wrong","std_student_acc_wrong","avg_student_acc_correct","discrimination_proxy"]: Xb[c] = pf[c].values
for c in Xb: Xb[c] = Xb[c].fillna(Xb[c].median())
for c in ["p_wrong_option","p_correct","n_attempts","n_wrong_option"]: Xb[c] = np.log1p(Xb[c])
Xb = StandardScaler().fit_transform(Xb.to_numpy())
for k in (6, 8): PART[f"Behavioural (k={k})"] = KMeans(k, random_state=SEED, n_init=20).fit_predict(Xb)

# ---------------- statistics
def cov(lab, fam):
    K = lab.max() + 1; ct = np.bincount(lab * 3 + fam, minlength=K * 3).reshape(K, 3); n = ct.sum(1)
    return float((ct.max(1)[n > 0] / n[n > 0]).mean())
STRATA = {"L2": codes(L2), "leaf": codes(LEAF)}
GROUPS = {id(v): [np.where(v == s)[0] for s in np.unique(v)] for v in STRATA.values()}
def strata_perm(lab, strata, rng):
    out = lab.copy()
    for i in GROUPS[id(strata)]:
        if len(i) > 1: out[i] = lab[rng.permutation(i)]
    return out
rows = []
for pname, lab in PART.items():
    lab = codes(lab); K = int(lab.max() + 1)
    for rname, fam in FAM.items():
        obs = cov(lab, fam); rng = np.random.default_rng(SEED)
        null = np.array([cov(rng.permutation(lab), fam) for _ in range(NPERM)])
        row = dict(partition=pname, k=K, retriever=rname, coverage=obs, null_free=null.mean(), p_free=float((null >= obs).mean()))
        if not pname.startswith("Eedi"):
            for sname, st in STRATA.items():
                rng = np.random.default_rng(SEED); nl = np.array([cov(strata_perm(lab, st, rng), fam) for _ in range(NPERM)])
                row[f"null_within_{sname}"] = nl.mean(); row[f"p_within_{sname}"] = float((nl >= obs).mean())
        mi = adjusted_mutual_info_score(TOP1[rname], lab); row["AMI_misconception_id"] = mi
        if not pname.startswith("Eedi"):
            rng = np.random.default_rng(SEED); nl = np.array([adjusted_mutual_info_score(TOP1[rname], strata_perm(lab, STRATA["L2"], rng)) for _ in range(200)])
            row["AMI_null_within_L2"] = nl.mean(); row["AMI_p_within_L2"] = float((nl >= mi).mean())
        rows.append(row)
    print(f"{pname:55s} k={K:3d} " + " | ".join(f"{r['retriever']}: {r['coverage']:.3f} (free {r['null_free']:.3f}"
          + (f", within-L2 {r['null_within_L2']:.3f} p={r['p_within_L2']:.3f}, within-leaf {r['null_within_leaf']:.3f} p={r['p_within_leaf']:.3f}" if 'null_within_L2' in r else "") + ")"
          for r in rows[-2:]), flush=True)
R = pd.DataFrame(rows); R.to_csv(RESULTS / "topic_baseline.csv", index=False)

# ---------------- retrieval-free check on annotated 2024 items
Mt = pd.read_csv(RESULTS / "match_2024_links.csv")
link = Mt[Mt.letter.notna() & Mt.correct_match & (Mt.sim >= 0.80) & Mt.mis.notna()].copy()
link["mis"] = link.mis.astype(int); idx = link.idx.to_numpy(); ann = link.mis.to_numpy()
ann_fam = bank_fam[ann]
chk = dict(n_linked=int((Mt.letter.notna() & Mt.correct_match & (Mt.sim >= 0.80)).sum()), n_labelled=int(len(link)),
           n_questions=int(link.q24.nunique()), n_distinct_misconceptions=int(len(set(ann))))
for rname, S in SIMS.items():
    rank = (S[idx] > S[idx, ann][:, None]).sum(1)
    chk[f"{rname}_top1_exact"] = float((rank == 0).mean()); chk[f"{rname}_top25_exact"] = float((rank < 25).mean())
    chk[f"{rname}_family_agreement"] = float((FAM[rname][idx] == ann_fam).mean())
chk["annotated_family_share"] = np.bincount(ann_fam, minlength=3).tolist()
pairs = []
ii, jj = np.triu_indices(len(idx), 1)
same_mis = ann[ii] == ann[jj]; same_fam = ann_fam[ii] == ann_fam[jj]
for pname, lab in PART.items():
    l = np.asarray(lab)[idx]; same = l[ii] == l[jj]
    pairs.append(dict(partition=pname, n_same_pairs=int(same.sum()),
                      p_same_misconception_given_same_group=float(same_mis[same].mean()) if same.any() else np.nan,
                      p_same_family_given_same_group=float(same_fam[same].mean()) if same.any() else np.nan,
                      p_same_misconception_given_diff_group=float(same_mis[~same].mean()),
                      recall_same_misconception_pairs=float(same[same_mis].mean())))
P = pd.DataFrame(pairs); P.to_csv(RESULTS / "annotated_pairs.csv", index=False)
chk["base_rate_same_misconception"] = float(same_mis.mean()); chk["base_rate_same_family"] = float(same_fam.mean())
json.dump(chk, open(RESULTS / "annotated_check.json", "w"), indent=1)
print(json.dumps(chk, indent=1)); print(P.round(3).to_string(index=False))
