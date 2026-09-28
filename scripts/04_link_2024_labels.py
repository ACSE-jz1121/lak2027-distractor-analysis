"""Link the 899 NeurIPS-2020 distractors to the human-annotated Eedi 2024 competition items (train.csv).
A 2020 distractor is linked when (i) its question + correct answer is the nearest 2024 item by character
TF-IDF after LaTeX/whitespace normalisation, (ii) the correct answers are identical after normalisation and
(iii) the distractor text equals one of the 2024 wrong options. The linked option's MisconceptionId (annotated
by Eedi) gives a retrieval-free misconception label. Output -> results/match_2024_links.csv (row indices and 2024 ids only)"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (EDM, TRAIN_CSV, META, IMAGES, DISTRACTORS, MISC_ZIP, RESULTS, FIGURES, OUTPUTS, ASSETS,
                    SEED as CFG_SEED, GROUP_ENCODER, GROUP_K, emb)
import zipfile, io, re
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

z = zipfile.ZipFile(MISC_ZIP)
tr = pd.read_csv(io.BytesIO(z.read("train.csv")))
df = pd.read_parquet(DISTRACTORS).reset_index(drop=True)
ex = df.text_input.str.extract(r"Question:\s*(?P<q>.*?)\nCorrect:\s*(?P<c>.*?)\nWrong:\s*(?P<w>.*)", flags=re.S)

FRAC = re.compile(r"\\frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}")
def norm(s):
    s = str(s).lower()
    s = FRAC.sub(r"\1/\2", s)
    s = s.replace("\\times", "x").replace("\\div", "/").replace("×", "x").replace("÷", "/").replace("−", "-").replace("–", "-")
    s = re.sub(r"\\[a-z]+", " ", s)
    return re.sub(r"[^a-z0-9./x\-+=%]", "", s)

A = ex.q.map(norm) + "|" + ex.c.map(norm)
tr["ctext"] = [r[f"Answer{r.CorrectAnswer}Text"] for _, r in tr.iterrows()]
B = tr.QuestionText.map(norm) + "|" + tr.ctext.map(norm)
v = TfidfVectorizer(analyzer="char", ngram_range=(3, 5)).fit(pd.concat([A, B]))
S = (v.transform(A) @ v.transform(B).T).toarray(); j = S.argmax(1); s = S.max(1)
rows = []
for i in range(len(df)):
    r = tr.iloc[j[i]]; w = norm(ex.w[i]); hit = None
    for L in "ABCD":
        if L != r.CorrectAnswer and norm(r[f"Answer{L}Text"]) == w and w: hit = L
    rows.append(dict(idx=i, sim=float(s[i]), q24=int(r.QuestionId), letter=hit, correct_match=norm(r.ctext) == norm(ex.c[i]),
                     mis=r[f"Misconception{hit}Id"] if hit else np.nan))
M = pd.DataFrame(rows)
M["linked"] = M.letter.notna() & M.correct_match & (M.sim >= 0.5)
for t in [0.9, 0.8, 0.7, 0.6, 0.5]:
    m = (M.sim >= t) & M.letter.notna() & M.correct_match
    print(t, int(m.sum()), "with label:", int((m & M.mis.notna()).sum()))
M.to_csv(RESULTS / "match_2024_links.csv", index=False)
T = tr.set_index("QuestionId").QuestionText
for _, r in M[M.letter.notna() & M.correct_match].sort_values("sim").head(10).iterrows():
    print(round(r.sim, 2), "|", ex.q[r.idx][:70].replace("\n", " "), "||", T[r.q24][:70].replace("\n", " "))
