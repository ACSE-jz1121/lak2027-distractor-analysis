"""Candidate Error Review Workspace (educator-facing GUI) for the Qwen3-Embedding-8B candidate groups (k = 8).

Data sources (nothing here re-runs the analysis):
  outputs/audit.json                       candidate audit: topics, family evidence, Jaccard, held-out, verdicts (step 6)
  outputs/selector_quartiles.csv, outputs/confidence.csv   per-distractor learner data (steps 7-8)
  results/group_behaviour.csv              number of distractors with confidence ratings (step 9)
  results/who_makes_each_error.csv         proficiency sensitivity, recurrence, confidence (step 12)
  results/proficiency_curves.csv           selection by proficiency decile (step 10)
  assets/review_interpretations.json       interpretation + evidence-linked claims per group, written from the evidence
        packs of `13_review_workspace.py packs` (representative distractors + learner-response summaries only;
        no external misconception-bank information)
  assets/review_unconstrained.json         OPTIONAL unconstrained summaries for the evaluation mode
  assets/review_workspace_template.html    page template (plain HTML/CSS/JS; the data are injected as JSON)
Representative distractors = the 15 distractors nearest the group centroid (Qwen3-Embedding-8B, standardised, k-means
k = 8, seed 42), ids e1..e15 in order of distance.

Usage:  python scripts/13_review_workspace.py packs   -> writes the interpretation evidence packs (outputs/review_evidence_packs/)
        python scripts/13_review_workspace.py         -> builds interface/review_workspace.html
"""
import io, json, base64, ast, re
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (EDM, TRAIN_CSV, META, IMAGES, DISTRACTORS, RESULTS, FIGURES, OUTPUTS, ASSETS, ROOT,
                    SEED as CFG_SEED, GROUP_ENCODER, GROUP_K, emb)
from collections import Counter
import numpy as np, pandas as pd
from PIL import Image
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

SEED, K, NEX = 42, 8, 15
IMG = IMAGES
A = json.load(open(OUTPUTS / "audit.json", encoding="utf-8"))
df = pd.read_parquet(DISTRACTORS).reset_index(drop=True); N = len(df)
X = StandardScaler().fit_transform(emb(GROUP_ENCODER)).astype(np.float32)
km = KMeans(K, random_state=SEED, n_init=20).fit(X); lab, cen = km.labels_, km.cluster_centers_
assert [int((lab == c).sum()) for c in range(K)] == [c["n"] for c in A["candidates"]]
sm = pd.read_csv(META / "subject_metadata.csv", encoding="utf-8-sig")
qm = pd.read_csv(META / "question_metadata_task_3_4.csv", encoding="utf-8-sig")
lvl, nm = dict(zip(sm.SubjectId, sm.Level)), dict(zip(sm.SubjectId, sm.Name))
df["topic"] = df.QuestionId.map(qm.set_index("QuestionId").SubjectId.apply(ast.literal_eval)).apply(lambda s: next((nm[i] for i in s if lvl.get(i) == 2), "(none)"))
QQ = pd.read_csv(OUTPUTS / "selector_quartiles.csv").set_index("sample"); CF = pd.read_csv(OUTPUTS / "confidence.csv").set_index("sample")
sid = lambda i: f"{int(df.QuestionId.iloc[i])}_row{i}"
clean = lambda s: " ".join(str(s).split())

def central(c):
    idx = np.where(lab == c)[0]
    return idx[np.argsort(((X[idx] - cen[c]) ** 2).sum(1))][:NEX]

def exemplar(i, k):
    r = df.iloc[i]; q = QQ.loc[sid(i)]; cf = CF.loc[sid(i)]
    return dict(id=f"e{k}", sample=sid(i), question=clean(r.llm_question), correct=clean(r.correct_text), distractor=clean(r.distractor_text),
                options={L: clean(r[f"llm_{L}"]) for L in "ABCD"}, topic=r.topic,
                selection=round(float(r.p_wrong_option), 3), error_rate=round(float(r.difficulty), 3), attempts=int(r.n_attempts),
                weakest=round(float(q.share_Q1), 3), strongest=round(float(q.share_Q4), 3),
                sel_conf=None if pd.isna(cf.sel_conf) or cf.n_conf < 20 else round(float(cf.sel_conf)),
                corr_conf=None if pd.isna(cf.corr_conf) or cf.n_conf < 20 else round(float(cf.corr_conf)),
                confident_error=bool(cf.confident_error))

if len(sys.argv) > 1 and sys.argv[1] == "packs":
    out = OUTPUTS / "review_evidence_packs"; out.mkdir(exist_ok=True)
    W = pd.read_csv(RESULTS / "who_makes_each_error.csv").set_index("group")
    for cnd in A["candidates"]:
        c = int(cnd["id"][1:]); ex = [exemplar(int(i), k + 1) for k, i in enumerate(central(c))]
        w = W.loc[cnd["id"]]
        pack = dict(candidate=cnd["id"], n_distractors=cnd["n"],
                    note="Interpretation evidence only: representative distractors and learner-response summaries. "
                         "No external misconception-bank information is included.",
                    learner_response_summary=dict(selected_overall=round(float(w.chosen_all), 3), selected_weakest_quarter=round(float(w.chosen_q1), 3),
                                                  selected_strongest_quarter=round(float(w.chosen_q4), 3), or_per_sd_adjusted=round(float(w.or_adj), 3),
                                                  recurrence_all=round(float(w.spec_all), 3), recurrence_incorrect=round(float(w.spec_wrong), 3),
                                                  confident_error_share=None if pd.isna(w.confident_error) else round(float(w.confident_error), 3)),
                    representative_distractors=[{k: v for k, v in e.items() if k not in ("options",)} for e in ex])
        json.dump(pack, open(out / f"{cnd['id']}.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
        print(f"\n### {cnd['id']}  n={cnd['n']}")
        for e in ex:
            print(f"  {e['id']:>4} [{e['topic'][:22]}] {e['question'][:150]} | correct: {e['correct'][:40]} | chosen: {e['distractor'][:40]} | sel {e['selection']:.2f} err {e['error_rate']:.2f} weak {e['weakest']:.2f} strong {e['strongest']:.2f} conf {e['sel_conf']}/{e['corr_conf']}{' CONF-ERR' if e['confident_error'] else ''}")
    sys.exit()

# ============================================================== page data
TITLES = {"S0": "Negative-number sign errors", "S1": "Fraction representation and equivalence errors",
          "S2": "Mixed symmetry, shape-property and construction errors", "S3": "Mixed arithmetic and operation errors",
          "S4": "Area and perimeter errors", "S5": "Coordinate order, sign and geometric transformation errors",
          "S6": "Visual estimation of angle size", "S7": "Confusion between factors and multiples"}
SHORT = {"S0": "Negative-number signs", "S1": "Fraction representation", "S2": "Mixed shapes/symmetry", "S3": "Mixed arithmetic",
         "S4": "Area and perimeter", "S5": "Coordinates/transformation", "S6": "Angle estimation", "S7": "Factors vs multiples"}
INDICATE = {  # cautious educational reading of the behavioural evidence (observational; no intervention claims)
 "S0": "The same learners tend to show these sign errors across different quizzes, also when only incorrect responses are considered. This pattern may warrant closer inspection for learners who repeatedly select similar distractors.",
 "S1": "Selection falls away quickly as proficiency rises, with little evidence that particular learners repeat it. The pattern appears closely tied to general proficiency rather than to a persistent pattern in specific learners.",
 "S2": "This group mixes several error forms. They remain relatively common among stronger learners and are often selected with confidence, so the separate forms may be worth reviewing individually.",
 "S3": "This group mixes several error forms. The same learners tend to answer these items incorrectly, but not specifically by choosing these distractors, so the recurrence may reflect general difficulty with these items.",
 "S4": "This error remains relatively common among stronger learners and is often selected with confidence, without recurring in particular learners. This suggests a broader conceptual difficulty that may warrant review.",
 "S5": "The same learners tend to make coordinate errors across quizzes, and about a third of these distractors are selected with confidence. The recurrence is less specific among incorrect responses, so it may partly reflect general difficulty with coordinate items.",
 "S6": "The same learners tend to make angle errors across quizzes, but the recurrence is weaker among incorrect responses only. The pattern may partly reflect a general tendency to answer these items incorrectly.",
 "S7": "More concentrated among lower-proficiency learners and shows learner-specific recurrence across quizzes. This pattern may warrant closer inspection for learners who repeatedly select similar distractors.",
}
PROFILE = {  # descriptive summaries of observed behaviour (not learner diagnoses)
 "S0": ("Learner-specific recurring error",
        "The same learners tended to make this error across separate quizzes, also when only incorrect responses are considered."),
 "S1": ("Strongly proficiency-linked, weakly recurrent",
        "Selection declines steeply as proficiency increases, with little evidence that the same learners make this error repeatedly."),
 "S2": ("Broad-proficiency, high-confidence mixed pattern",
        "Selection stays comparatively common among stronger learners and is often confident, but the group itself is not coherent."),
 "S3": ("Proficiency-linked, with limited error-specific recurrence",
        "The same learners tended to answer these items wrongly, but not by choosing these particular distractors."),
 "S4": ("Broad-proficiency, relatively high-confidence pattern",
        "This error remains comparatively common among stronger learners and is relatively often selected with confidence, but shows little evidence of learner-specific recurrence."),
 "S5": ("Moderately learner-specific pattern",
        "Errors in this group recur within the same learners across quizzes, but the particular distractor chosen is less consistent."),
 "S6": ("Recurring pattern, weaker error-specific recurrence",
        "The same learners tended to make errors in this group across quizzes; the recurrence is weaker among incorrect responses only."),
 "S7": ("Proficiency-sensitive with learner-specific recurrence",
        "This pattern declines strongly with proficiency and also recurs within the same learners across quizzes."),
}
STATUS = {"Supported candidate": "supported", "Partly supported": "partial", "Not a coherent misconception candidate": "notcoherent"}
FRAC = re.compile(r"\\frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}")
def tex(s): return FRAC.sub(r"\1/\2", s).replace("\\times", "×").replace("\\div", "÷")

def img64(i, size=560):
    p = IMG / f"{int(df.QuestionId.iloc[i])}.jpg"
    if not p.exists(): return ""
    im = Image.open(p).convert("RGB"); im.thumbnail((size, size)); b = io.BytesIO(); im.save(b, "JPEG", quality=70)
    return "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()

W = pd.read_csv(RESULTS / "who_makes_each_error.csv").set_index("group")
PC = pd.read_csv(RESULTS / "proficiency_curves.csv")
GB = pd.read_csv(RESULTS / "group_behaviour.csv").set_index("group")
INT = json.load(open(ASSETS / "review_interpretations.json", encoding="utf-8"))
UNC_P = ASSETS / "review_unconstrained.json"
UNC = json.load(open(UNC_P, encoding="utf-8")) if UNC_P.exists() else None

def level_prof(o): return "strong" if o <= 0.60 else "moderate" if o <= 0.66 else "weaker"
def level_rec(m, lo): return "clear" if lo > 0 and m >= 0.075 else "some" if lo > 0 else "little"
def level_spec(m, lo): return "persists" if lo > 0 else "weaker" if m >= 0.03 else "none"
def level_conf(c): return "insufficient" if pd.isna(c) else "high" if c >= 0.30 else "moderate" if c >= 0.18 else "low"

groups, problems = [], []
for cnd in A["candidates"]:
    g = cnd["id"]; c = int(g[1:]); idx = central(c); ex = [exemplar(int(i), k + 1) for k, i in enumerate(idx)]
    for e, i in zip(ex, idx):
        e["question"] = tex(e["question"]); e["correct"] = tex(e["correct"]); e["distractor"] = tex(e["distractor"])
        e["img"] = img64(int(i))
    it = INT[g]; valid = {e["id"] for e in ex}
    for k, cl in enumerate(it["claims"]):
        bad = [x for x in cl["examples"] if x not in valid]
        if not cl["examples"] or bad: problems.append((g, k, bad))
    w = W.loc[g]; fx = cnd["family"]; st = STATUS[cnd["audit"]["verdict"]]
    top_share = cnd["topics"][0]["share"]; tpl = cnd["template_share"]
    ext = "strong" if fx["p_ex"] < 0.05 and fx["p_all"] < 0.05 else "moderate" if (fx["p_ex"] < 0.05 or fx["p_all"] < 0.05) else "weak"
    checks = [dict(key="stability", label="Stable across bootstrap samples", value=f"Bootstrap Jaccard {cnd['stability']['jaccard']:.2f}",
                   rule="Jaccard ≥ 0.60", ok=bool(cnd["stability"]["jaccard"] >= 0.60), partial=False),
              dict(key="topic", label="Coherent representative core", value=f"Dominant topic ({cnd['topics'][0]['topic']}): {100*top_share:.1f}%",
                   rule="dominant topic ≥ 50% (≥ 40% counts towards partial support)", ok=bool(top_share >= 0.50), partial=bool(top_share >= 0.40)),
              dict(key="family", label="External misconception-family evidence", value=f"Core-15 family concentration {100*fx['share_ex']:.0f}% (p = {fx['p_ex']:.3f})",
                   rule="core-15 family above same-size random groups, p < 0.05", ok=bool(fx["p_ex"] < 0.05), partial=bool(fx["p_all"] < 0.05)),
              dict(key="format", label="Not driven by a question format", value=f"Two-speaker template share {100*tpl:.0f}%",
                   rule="one template < 60% of distractors", ok=bool(tpl < 0.60), partial=False)]
    curve = PC[PC.group == g].sort_values("dec").p_sel.round(4).tolist()
    groups.append(dict(
        id=g, title=TITLES[g], short=SHORT[g], status=st, n=cnd["n"], template=round(tpl, 3), jaccard=round(cnd["stability"]["jaccard"], 2), ext=ext,
        profile=PROFILE[g][0], profile_text=PROFILE[g][1], indicate=INDICATE[g], checks=checks,
        topics=[dict(t=t["topic"], s=round(t["share"], 3)) for t in cnd["topics"][:3]],
        external=dict(family=fx["dominant"], share_all=round(fx["share_all"], 3), null_all=round(fx["null_all"], 3), p_all=fx["p_all"],
                      share_ex=round(fx["share_ex"], 3), null_ex=round(fx["null_ex"], 3), p_ex=fx["p_ex"],
                      matches=[dict(name=m["name"], count=m["count"]) for m in cnd["misconceptions"][:5]]),
        heldout=dict(n=cnd["heldout"]["n_assigned"], agree=cnd["heldout"]["mean_agreement"]),
        beh=dict(all=round(float(w.chosen_all), 3), q1=round(float(w.chosen_q1), 3), q4=round(float(w.chosen_q4), 3), or_adj=round(float(w.or_adj), 3),
                 rec=round(float(w.spec_all), 4), rec_lo=round(float(w.spec_all_lo), 4), rec_hi=round(float(w.spec_all_hi), 4),
                 spec=round(float(w.spec_wrong), 4), spec_lo=round(float(w.spec_wrong_lo), 4), spec_hi=round(float(w.spec_wrong_hi), 4),
                 conf=None if pd.isna(w.confident_error) else round(float(w.confident_error), 3),
                 conf_rated=int(GB.loc[g, "conf_rated"]), curve=curve),
        levels=dict(prof=level_prof(w.or_adj), rec=level_rec(w.spec_all, w.spec_all_lo), spec=level_spec(w.spec_wrong, w.spec_wrong_lo),
                    conf=level_conf(w.confident_error)),
        interp=it, unconstrained=(UNC or {}).get(g), exemplars=ex))
assert not problems, f"citation audit failed: {problems}"
data = dict(groups=groups, generated_by=INT["_meta"]["generator"], has_unconstrained=UNC is not None,
            or_range=[min(x["beh"]["or_adj"] for x in groups), max(x["beh"]["or_adj"] for x in groups)])
tpl_html = (ASSETS / "review_workspace_template.html").read_text(encoding="utf-8")
html = tpl_html.replace("__DATA__", json.dumps(data, ensure_ascii=False).replace("</", "<\\/"))
OUT_HTML = ROOT / "interface" / "review_workspace.html"; OUT_HTML.parent.mkdir(exist_ok=True)
OUT_HTML.write_text(html, encoding="utf-8")
print("review_workspace.html", len(html) // 1024, "KB; claims per group:", {g["id"]: len(g["interp"]["claims"]) for g in groups})
