# Auditable candidate misconceptions from MCQ response logs

Code for the LAK 2027 submission on turning multiple-choice distractors into **candidate misconception groups** that
educators can audit. Semantic and behavioural representations of the same distractors play complementary roles:

* **semantic clustering** of the item and distractor text proposes candidate misconception groups;
* a **rule-based audit** decides which groups are carried forward as candidates;
* **learner-response evidence** describes how each group appears across learners: how it changes with proficiency,
  whether it recurs in the same learners across quizzes, and how often it is a confident error;
* an **educator-facing review interface** combines an LLM interpretation, in which every claim cites a numbered
  representative distractor, with the representative distractors and the learner-response summaries.

## Repository layout

```
config.py            paths and constants (data location via MCQ_DATA_DIR)
run_all.sh           runs every step in order
scripts/             01-14: analysis pipeline and interface builder (see table below)
scripts/figures/     figure scripts and the TikZ source of the pipeline figure
assets/              candidate titles, LLM interpretations with evidence links, interface template
interface/           review_workspace.html: the educator-facing review interface (self-contained, open in a browser)
data/                input data (not included, see data/README.md)
```

Running the pipeline creates `cache/` (embeddings), `results/` (result tables), `figures/` (figures) and `outputs/`
(item-level files that contain Eedi content). None of these are part of the repository.

## Review interface

[`interface/review_workspace.html`](interface/review_workspace.html) is a single self-contained file (no server, no
installation): download it and open it in a browser. It has three views:

* **Review**: one candidate group at a time: audit status, generated interpretation with evidence-linked claims
  (hover or click an example ID to highlight the cited distractor), representative distractors, how the error appears
  across learners, and collapsible candidate-audit and external-evidence panels. External misconception-bank evidence
  is shown only in its own panel; it is not part of the evidence given to the LLM interpretation.
* **Compare groups**: all candidate groups positioned by proficiency sensitivity and learner-specific recurrence,
  with a table of the group-level statistics.
* **Evaluation mode**: the card layout used for rating, with five-point scales and JSON/CSV export (ratings are kept
  in the browser's local storage only).

A group or view can be opened directly with a URL fragment, e.g. `review_workspace.html#g=S7&v=review`,
`#g=S7&v=compare` or `#g=S7&v=eval`.

The page embeds the 15 representative distractors of each group (question text, answer options and a small question
image) from the Eedi release. It is rebuilt by `scripts/13_review_workspace.py`.

## Setup

```bash
pip install -r requirements.txt
```

Put the data under `data/` as described in [`data/README.md`](data/README.md). A GPU helps with encoding; on an 8 GB
card Qwen3-Embedding-4B and 8B fall back to CPU (about 15 and 30 minutes).

## Reproducing the results

```bash
bash run_all.sh            # or run the scripts one by one from the repository root
```

| Step | Script | Produces |
|---|---|---|
| 1 | `scripts/01_encode.py` | embeddings for 10 encoders and the misconception bank (`cache/embeddings/`) |
| 2 | `scripts/02_encoder_comparison.py` | tuning grid, stability, topic NMI, cross-view ARI, external alignment and held-out reliability for every encoder: `results/encoder_comparison.csv`, `encoder_tuning_grid.csv`, `encoder_pairwise_ari_k8.csv` |
| 3 | `scripts/03_weight_path.py` | secondary analysis: combined semantic-behavioural representations along a weight path: `results/weight_path.csv` |
| 4 | `scripts/04_link_2024_labels.py` | links distractors to human-annotated items of the 2024 Eedi competition: `results/match_2024_links.csv` |
| 5 | `scripts/05_topic_baseline.py` | topic-tag baseline, within-topic permutation test and the annotated-subset check: `results/topic_baseline.csv`, `annotated_pairs.csv`, `annotated_check.json` |
| 6 | `scripts/06_candidate_audit.py` | candidate groups (Qwen3-Embedding-8B, k = 8), rule-based audit, held-out assignment: `results/candidate_audit.csv` |
| 7 | `scripts/07_selector_quarters.py` | ability-quarter composition of each distractor's selectors |
| 8 | `scripts/08_confidence.py` | learner confidence of selectors and correct responders |
| 9 | `scripts/09_group_behaviour.py` | learner-response behaviour per candidate group and Kruskal-Wallis tests: `results/group_behaviour.csv`, `group_behaviour_tests.csv`, `group_behaviour_table.tex` |
| 10 | `scripts/10_proficiency.py` | selection by learner proficiency: per-distractor logistic slopes, difficulty-adjusted odds ratios per group, decile curves: `results/proficiency_*.csv`, `figures/proficiency_by_group.png` |
| 11 | `scripts/11_learner_recurrence.py` | learner-specific recurrence across quizzes (quiz-level split halves, all attempts and incorrect responses only): `results/learner_consistency.csv`, `figures/learner_consistency.png` |
| 12 | `scripts/12_who_table.py` | "who makes each error" summary per group: `results/who_makes_each_error.csv`, `who_makes_each_error_table.tex` |
| 13 | `scripts/13_review_workspace.py packs` | interpretation evidence packs (representative distractors and learner-response summaries, no external evidence) used to write `assets/review_interpretations.json` |
| 13 | `scripts/13_review_workspace.py` | the review interface: `interface/review_workspace.html` |
| 14 | `scripts/14_learner_patterns.py` | learner-pattern figure and tables: `figures/learner_patterns.png`, `results/learner_patterns_table.tex`, `learner_patterns_interpretation_table.tex` |
| – | `scripts/figures/*.py`, `fig1_pipeline.tex` | `figures/*.png` |

Steps 7–8 and 13 write item-level files to `outputs/` because they contain Eedi question text; these files are not
committed.

### Audit rule

Each semantic group gets a transparent, rule-based status (`06_candidate_audit.py`):

* content evidence is **strong** when one Eedi level-2 topic covers a majority (≥ 50%) of its distractors and its 15
  most central distractors fall in one misconception family more often than same-size random groups (p < 0.05), and
  **moderate** when the dominant topic covers ≥ 40% or the family alignment over all members exceeds chance;
* groups in which one two-speaker item template covers ≥ 60% of distractors are treated as format-driven;
* groups with a mean bootstrap Jaccard below 0.60 are unstable (Hennig, 2007);
* **supported** = strong and not unstable; **partly supported** = moderate and not unstable; otherwise **not coherent**.

Learner-response and held-out evidence is shown in the interface but does not enter the status. The status is
unchanged for majority thresholds between 45% and 55%.

### Evidence-linked interpretations

`assets/review_interpretations.json` holds, for each candidate group, a short interpretation and 3–5 claims, each
citing the representative distractors (e1–e15, the 15 distractors nearest the group centroid) that support it. They
were written by an LLM from the evidence packs of step 13 only: representative distractors and learner-response
summaries, without any external misconception-bank information. `13_review_workspace.py` checks that every claim cites
at least one valid example and stops otherwise. The interpretations and `assets/titles.json` belong to the
Qwen3-Embedding-8B candidates; if you change `GROUP_ENCODER` in `config.py`, regenerate them from the new evidence packs.

For the evaluation mode, unconstrained summaries (same structure, without citations) can be added as
`assets/review_unconstrained.json`; without that file the unconstrained condition shows a notice.

## Notes

* All random steps use seed 42; the question-level split is 70/10/20 (629 / 90 / 180 distractors).
* Re-encoding can change embeddings at the level of 1e-7 (GPU floating point), which is enough to move k-means
  bootstrap stability in the third decimal (e.g. MiniLM at k = 8: 0.751 with the paper's embeddings, 0.746 after
  re-encoding). All other numbers were reproduced exactly from the cached embeddings. For exact reproduction, use the
  embedding cache attached to the GitHub release (unpack into `cache/embeddings/`) and skip step 1.
* The combined-feature weight path (`03_weight_path.py`) is a secondary analysis: with every encoder, the combined
  partition moves from the semantic to the behavioural partition as the behavioural weight grows, without forming a
  new stable joint structure.
* `gte-large-en-v1.5` was tried but excluded: its remote modelling code is incompatible with transformers ≥ 5 and gives
  collapsed embeddings.
* The retrieved misconception descriptions are used only at the level of three broad families; on the 83 distractors
  that could be linked to human-annotated 2024 items, the exact description matched in about one case in ten, while the
  family agreed for 72% (MiniLM) and 83% (Qwen3-Embedding-0.6B).
* Learner-response summaries are observational descriptions of response patterns across learners; they are not
  diagnoses of individual learners' misconceptions.
