#!/usr/bin/env bash
# Run the full pipeline from the repository root. Stops at the first failing step.
set -euo pipefail
cd "$(dirname "$0")"
for s in 01_encode 02_encoder_comparison 03_weight_path 04_link_2024_labels 05_topic_baseline 06_candidate_audit \
         07_selector_quarters 08_confidence 09_group_behaviour 10_proficiency 11_learner_recurrence 12_who_table; do
  echo "=== $s"; python "scripts/$s.py"
done
echo "=== 13_review_workspace packs"; python scripts/13_review_workspace.py packs
echo "=== 13_review_workspace";       python scripts/13_review_workspace.py
echo "=== 14_learner_patterns";       python scripts/14_learner_patterns.py
for f in fig_encoders fig_topic_baseline fig_weight_path; do echo "=== $f"; python "scripts/figures/$f.py"; done
echo "done"
