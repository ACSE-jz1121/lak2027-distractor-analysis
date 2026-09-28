# What the Error Is and Who Makes It

Code and review interface for the LAK 2027 submission, *What the Error Is and Who Makes It: Auditable Misconception Analysis from Distractor Responses*.

The project groups incorrect options from mathematics multiple-choice questions by their content, checks whether those groups are coherent enough to interpret, and describes how the associated errors appear in learner responses. Each group is treated as a **candidate** misconception. The review interface lets educators inspect the examples and response patterns behind its interpretation.

## Start here

**Explore the results:** Open [`interface/review_workspace.html`](interface/review_workspace.html) in a browser. The page is self-contained and does not require the dataset, a server, or a Python environment.

**Run the analysis:** Install the requirements, obtain the Eedi data as described in [`data/README.md`](data/README.md), and run the pipeline from the repository root:

```bash
pip install -r requirements.txt
bash run_all.sh
```

The dataset is not included in this repository. `config.py` defines paths and constants, including the `MCQ_DATA_DIR` data location. A GPU speeds up the encoding step; Qwen3-Embedding-4B and 8B may fall back to CPU on an 8 GB GPU.

## What the analysis does

1. **Propose groups.** Encode each question, correct answer, and selected incorrect option, then cluster the distractors by semantic similarity. Behavioural features are analysed separately.
2. **Audit the groups.** Check topic concentration, external misconception-family alignment, resampling stability, and sensitivity to a repeated question template. The audit assigns each group a rule-based status.
3. **Describe learner responses.** Measure how selection varies with learner proficiency, whether the same learners select related distractors across quizzes, and how confident they are in those responses.
4. **Review the evidence.** Show representative distractors, learner-response summaries, and a short interpretation whose claims cite specific examples. The external misconception bank is used in the audit; it is not supplied to the interpretation generator.

The main analysis uses 899 distractors, one qualifying distractor per question. A separate question-level 70/10/20 split (629/90/180) tests assignments to groups learned from training data.

## Review interface

The self-contained [`review_workspace.html`](interface/review_workspace.html) has three views:

| View | What it shows |
| --- | --- |
| **Review** | One candidate group, its audit status, cited examples, learner-response summary, and expandable audit panels. Select an example citation to find the distractor behind a claim. |
| **Compare groups** | Candidate groups positioned by proficiency sensitivity and learner-specific recurrence, with group-level statistics. |
| **Evaluation mode** | The card layout used in the human evaluation, five-point rating scales, and JSON/CSV export. Ratings stay in the browser's local storage until exported. |

Links can open a particular group and view directly, for example `review_workspace.html#g=S7&v=review`, `#g=S7&v=compare`, or `#g=S7&v=eval`.

The page includes 15 representative distractors per group, with question text, answer options, and a small question image from the Eedi release. `scripts/13_review_workspace.py` builds it.

## Repository contents

| Path | Purpose |
| --- | --- |
| `config.py` | Paths, model settings, and analysis constants. |
| `run_all.sh` | Runs the analysis scripts in order. |
| `scripts/` | Encoding, comparisons, audit, behavioural analyses, and interface builder. |
| `scripts/figures/` | Figure scripts and the TikZ source of the pipeline figure. |
| `assets/` | Candidate titles, evidence-linked interpretations, and interface template. |
| `interface/review_workspace.html` | Browser-based review interface. |
| `data/README.md` | Instructions for obtaining and placing the input data. |

The pipeline writes embeddings to `cache/`, tables to `results/`, figures to `figures/`, and item-level files containing Eedi content to `outputs/`. These generated directories are not part of the repository.

## Pipeline and outputs

Run individual scripts from the repository root if you only need a particular analysis. `run_all.sh` runs them in order.

| Step | Script | Main result |
| --- | --- | --- |
| 1 | `scripts/01_encode.py` | Embeddings for ten encoders and the external misconception bank in `cache/embeddings/`. |
| 2 | `scripts/02_encoder_comparison.py` | Encoder tuning and comparisons, including stability, topic alignment, external alignment, and held-out reliability, in `results/encoder_*.csv`. |
| 3 | `scripts/03_weight_path.py` | Semantic–behavioural weighting analysis in `results/weight_path.csv`. |
| 4–5 | `scripts/04_link_2024_labels.py`, `05_topic_baseline.py` | Links to human-annotated 2024 items and topic-controlled comparisons in `results/`. |
| 6 | `scripts/06_candidate_audit.py` | Eight Qwen3-Embedding-8B candidate groups at $k=8$, their audit statuses, and held-out assignments in `results/candidate_audit.csv`. |
| 7–8 | `scripts/07_selector_quarters.py`, `08_confidence.py` | Ability-quarter composition and response confidence. Item-level files go to `outputs/`. |
| 9–12 | `scripts/09_group_behaviour.py` through `12_who_table.py` | Group behaviour, proficiency trends, learner-specific recurrence, and summary tables in `results/` and `figures/`. |
| 13 | `scripts/13_review_workspace.py packs`, then `scripts/13_review_workspace.py` | Evidence packs for the interpretations, then `interface/review_workspace.html`. |
| 14 | `scripts/14_learner_patterns.py` | Learner-pattern figure and tables. |

Additional figure scripts are in `scripts/figures/`. Steps 7–8 and 13 also produce item-level Eedi files under `outputs/`.

### How the candidate audit works

`scripts/06_candidate_audit.py` assigns a status using these checks:

- **Strong content evidence:** one Eedi level-2 topic accounts for at least 50% of the group, and the 15 most central distractors align with one external misconception family more often than same-size random groups ($p<0.05$).
- **Moderate content evidence:** the leading topic accounts for at least 40%, or misconception-family alignment across all group members exceeds chance.
- **Format effect:** one two-speaker item template accounts for at least 60% of the group.
- **Instability:** mean bootstrap Jaccard is below 0.60.

A group is **supported** if it has strong content evidence and is stable; **partly supported** if it has moderate content evidence and is stable; otherwise it is **not coherent**. The interface also displays learner-response and held-out evidence, which do not determine this status. Changing the majority threshold within 45–55% does not change the reported statuses.

### Evidence-linked interpretations

`assets/review_interpretations.json` contains a short interpretation and three to five claims per group. Each claim cites at least one of the 15 distractors closest to that group's centroid, labelled `e1`–`e15`. The interpretations were written from the evidence packs produced by step 13, which contain examples and learner-response summaries but no external misconception-bank evidence. The interface builder checks every citation and stops if a claim cites an invalid example.

The supplied interpretations and `assets/titles.json` refer to the Qwen3-Embedding-8B groups. If you change `GROUP_ENCODER` in `config.py`, generate new evidence packs and interpretations for the new groups before rebuilding the interface.

Evaluation mode can also show unconstrained summaries from `assets/review_unconstrained.json`. If that optional file is absent, the interface displays a notice for that condition.

## Reproducibility notes

- Random steps use seed 42.
- Small floating-point differences during GPU encoding can affect k-means bootstrap stability in the third decimal. For example, MiniLM at $k=8$ gave 0.751 with the paper's embeddings and 0.746 after re-encoding. Other reported numbers were reproduced from the cached embeddings. If the paper's embedding cache is available with the release, unpack it into `cache/embeddings/` and skip step 1 for an exact match.
- The weighting path in step 3 is a secondary analysis. It did not produce a separate, consistently stable joint grouping between the semantic and behavioural endpoints.
- `gte-large-en-v1.5` was tested but excluded because its remote modelling code was incompatible with `transformers` version 5 and above and produced collapsed embeddings.
- External misconception descriptions are compared at the level of three broad families. On the 83 distractors linked to human-annotated 2024 items, exact descriptions matched in roughly one case in ten; family agreement was 72% with MiniLM and 83% with Qwen3-Embedding-0.6B.

Learner-response summaries describe observed patterns across learners. They do not diagnose an individual learner's misconception.
