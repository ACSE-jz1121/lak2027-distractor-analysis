# Data

The data are **not** included in this repository: both sources are released by Eedi under their own terms.
Place the files as below (or set `MCQ_DATA_DIR` to another folder with the same layout).

```
data/
├── eedi_2020/                                    NeurIPS 2020 Education Challenge release (tasks 3 and 4)
│   └── data/
│       ├── train_data/train_task_3_4.csv
│       ├── metadata/answer_metadata_task_3_4.csv
│       ├── metadata/question_metadata_task_3_4.csv
│       ├── metadata/subject_metadata.csv
│       └── images/<QuestionId>.jpg               (only needed for the audit cards)
├── distractor_tasks.parquet                      899 distractor instances (see below)
└── eedi-mining-misconceptions-in-mathematics.zip Kaggle "Eedi - Mining Misconceptions in Mathematics"
                                                  (misconception_mapping.csv and train.csv are read from the zip)
```

| File | Source |
|------|--------|
| `eedi_2020/` | NeurIPS 2020 Education Challenge, Eedi diagnostic questions (public release). |
| `eedi-mining-misconceptions-in-mathematics.zip` | Kaggle competition *Eedi – Mining Misconceptions in Mathematics* (2024). |
| `distractor_tasks.parquet` | Built from the 2020 release: one row per retained distractor (the most-selected wrong option of each question, 899 rows; keys `QuestionId`, `AnswerValue`, `CorrectAnswer`), with the behavioural features (`n_attempts`, `p_wrong_option`, `wrong_rank`, `option_entropy`, `difficulty`, `p_correct`, `n_wrong_option`, `avg_student_acc_wrong`, `std_student_acc_wrong`, `avg_student_acc_correct`, `discrimination_proxy`) and the question text parsed from the question images (`question_image_path`, `llm_question`, `llm_A`–`llm_D`, `correct_text`, `distractor_text`, and `text_input` = "Question: … / Correct: … / Wrong: …"). The text was extracted with an LLM from the question images; this step needs an API key and is described in the paper. |
