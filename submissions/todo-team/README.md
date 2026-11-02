# [TODO: Kaggle team name] — Human or AI?

Code for the CS5785 "Human or AI?" Kaggle competition, in two tracks that share one validation protocol:

| Folder | What | Report |
|---|---|---|
| [`pretrained/`](pretrained/) | Pretrained-LM features (GPT-2, Qwen2.5 Binoculars) and fine-tuned encoders (RoBERTa, DeBERTa-v3, OpenAI detector), blended | `pretrained/reports/kaggle_pretrained_report.tex` |
| [`classical/`](classical/) | Models trained from scratch, no pretrained weights: TF-IDF + LR, stylometric GBM, small Transformer | `classical/reports/kaggle_classical_report.tex` |
| [`common/`](common/) | Fixed validation split, data loading, metrics shared by both tracks | — |

Submission and testing history: [`SUBMISSIONS.md`](SUBMISSIONS.md).

## 1. Team

- Kaggle team name: `[TODO]`
- Members (name, NetID): `[TODO]`

## 2. Environment

Python 3.12. [TODO: per-track install commands]

Competition data is never committed. Put `train.csv`, `test.csv`, `unlabeled.csv` and `sample_submission.csv` in
`submissions/<team>/data/` (git-ignored), or point to them with `--data-dir PATH` or the `HOA_DATA_DIR` environment variable.

## 3. Validation protocol (train / evaluate)

```bash
cd submissions/<team>
python common/make_split.py          # verifies common/val_ids.csv is reproduced exactly
```

`common/val_ids.csv` holds the ids and labels of a fixed stratified 2,000-row validation set from `train.csv`
(1,000 human / 1,000 AI, `sklearn.model_selection.train_test_split`, seed `20261023`). Every model in both tracks
trains on the other 8,000 rows and is scored on these 2,000. Hyperparameters are chosen by cross-validation inside
the 8,000 rows, never on the validation set or on Kaggle. Final Kaggle models are refit on all 10,000 rows.

[TODO: per-track train/evaluate commands]

## 4. Command that writes `submission.csv`

[TODO]

## 5. Seeds

- Validation split and all new experiments: `20261023`.
- The original research runs in `pretrained/` used seed `42`. Where they are re-measured on the fixed split, the
  seed used is stated next to the result.

## 6. AI-tool disclosure

[TODO: summary matching report Q6.3]
