"""Who actually selects each distractor? Learners are split into quartiles of overall accuracy (all their answers,
learners with >= 20 answers). For each distractor we compare the quartile mix of its selectors with the quartile mix
of everyone who attempted the question (lift = share among selectors / share among attempters).
Output -> outputs/selector_quartiles.csv (per distractor)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (EDM, TRAIN_CSV, META, IMAGES, DISTRACTORS, MISC_ZIP, RESULTS, FIGURES, OUTPUTS, ASSETS,
                    SEED as CFG_SEED, GROUP_ENCODER, GROUP_K, emb)
import os
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

df = pd.read_parquet(DISTRACTORS).reset_index(drop=True)
raw = pd.read_csv(TRAIN_CSV, usecols=["QuestionId", "UserId", "AnswerValue", "IsCorrect"]).astype(int)
u = raw.groupby("UserId").IsCorrect.agg(["mean", "size"]); u = u[u["size"] >= 20]
u["q"] = pd.qcut(u["mean"], 4, labels=[1, 2, 3, 4]).astype(int)
cut = u.groupby("q")["mean"].agg(["min", "max"])
r = raw[raw.UserId.isin(u.index)].merge(u["q"], left_on="UserId", right_index=True)
att = r.groupby(["QuestionId", "q"]).size().unstack(fill_value=0); att = att.div(att.sum(1), axis=0)
sel = r.groupby(["QuestionId", "AnswerValue", "q"]).size().unstack(fill_value=0)
key = df[["QuestionId", "AnswerValue"]]
S = key.merge(sel, left_on=["QuestionId", "AnswerValue"], right_index=True, how="left").fillna(0)
n_sel = S[[1, 2, 3, 4]].sum(1); Sh = S[[1, 2, 3, 4]].div(n_sel, axis=0)
Ah = key[["QuestionId"]].merge(att, left_on="QuestionId", right_index=True, how="left")[[1, 2, 3, 4]]
lift = Sh.values / Ah.values
out = pd.DataFrame({"sample": df.QuestionId.astype(str) + "_row" + df.index.astype(str), "n_selectors": n_sel,
                    **{f"share_Q{k}": Sh[k].values for k in [1, 2, 3, 4]}, **{f"lift_Q{k}": lift[:, k - 1] for k in [1, 2, 3, 4]}, **{f"att_Q{k}": Ah[k].values for k in [1, 2, 3, 4]}})
lab = KMeans(GROUP_K, random_state=42, n_init=20).fit_predict(StandardScaler().fit_transform(emb(GROUP_ENCODER)))
out["candidate"] = [f"S{c}" for c in lab]
out.to_csv(OUTPUTS / "selector_quartiles.csv", index=False)

print("learner quartiles (overall accuracy):", {int(k): f"{v['min']:.2f}-{v['max']:.2f}" for k, v in cut.iterrows()}, "| learners:", len(u))
print("\nAll 899 distractors: median share of selectors in each quartile, and median lift vs attempters")
print(pd.DataFrame({"share": [out[f"share_Q{k}"].median() for k in [1, 2, 3, 4]], "lift": [out[f"lift_Q{k}"].median() for k in [1, 2, 3, 4]]}, index=["Q1 weakest", "Q2", "Q3", "Q4 strongest"]).round(2).to_string())
print(f"\nShare of distractors where the weakest quartile is over-represented (lift > 1): {(out.lift_Q1 > 1).mean():.0%}")
print(f"Share where the strongest quartile is over-represented (lift > 1): {(out.lift_Q4 > 1).mean():.0%}")
print(f"Share where Q4 learners make up >= 20% of selectors: {(out.share_Q4 >= 0.2).mean():.0%}")
print("\nBy candidate (medians):")
print(out.groupby("candidate")[["share_Q1", "share_Q2", "share_Q3", "share_Q4", "lift_Q1", "lift_Q4"]].median().round(2).to_string())
for s in ["251_row240", "406_row386", "558_row531", "634_row602"]:
    row = out[out["sample"] == s].iloc[0]
    print(s, {f"Q{k}": f"{row[f'share_Q{k}']:.0%} (x{row[f'lift_Q{k}']:.1f})" for k in [1, 2, 3, 4]})
