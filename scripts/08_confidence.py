"""Learner confidence for each distractor (Eedi answer metadata: self-rated confidence 0-100, ~25% of answers).
Per distractor: number of confidence ratings from its selectors, their mean confidence, the mean confidence of
learners who answered the same question correctly, the gap, and the share of selectors rating >= 75 ("confident").
A distractor is flagged as a 'confident error' when it has >= 20 ratings and selectors are within 5 points of
correct responders. Output -> outputs/confidence.csv"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (EDM, TRAIN_CSV, META, IMAGES, DISTRACTORS, MISC_ZIP, RESULTS, FIGURES, OUTPUTS, ASSETS,
                    SEED as CFG_SEED, GROUP_ENCODER, GROUP_K, emb)
import os
import numpy as np, pandas as pd

df = pd.read_parquet(DISTRACTORS).reset_index(drop=True)
tr = pd.read_csv(TRAIN_CSV, usecols=["QuestionId", "AnswerId", "AnswerValue", "IsCorrect"])
am = pd.read_csv(META / "answer_metadata_task_3_4.csv", usecols=["AnswerId", "Confidence"])
m = tr.merge(am, on="AnswerId", how="left").dropna(subset=["Confidence"])
sel = m.groupby(["QuestionId", "AnswerValue"]).Confidence.agg(n_conf="count", sel_conf="mean", sel_confident=lambda s: (s >= 75).mean())
cor = m[m.IsCorrect == 1].groupby("QuestionId").Confidence.agg(corr_conf="mean", corr_confident=lambda s: (s >= 75).mean(), n_corr="count")
out = df[["QuestionId", "AnswerValue"]].merge(sel, left_on=["QuestionId", "AnswerValue"], right_index=True, how="left") \
                                        .merge(cor, left_on="QuestionId", right_index=True, how="left")
out["n_conf"] = out.n_conf.fillna(0).astype(int)
out["conf_gap"] = out.corr_conf - out.sel_conf
out["enough"] = out.n_conf >= 20
out["confident_error"] = out.enough & (out.conf_gap <= 5)
out.insert(0, "sample", df.QuestionId.astype(str) + "_row" + df.index.astype(str))
out.to_csv(OUTPUTS / "confidence.csv", index=False)
ok = out[out.enough]
print(f"distractors with >= 20 ratings: {out.enough.mean():.0%} | selector confidence median {ok.sel_conf.median():.0f} "
      f"| correct responders {ok.corr_conf.median():.0f} | confident errors {ok.confident_error.mean():.0%} of rated distractors")
