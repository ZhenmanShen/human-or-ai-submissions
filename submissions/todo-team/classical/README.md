# Classical track: models trained from scratch

No pretrained weights, embeddings or tokenizers anywhere in this folder. Every model trains on the team's
fixed 8,000 rows and is scored on the same 2,000 validation rows as the pretrained track
(`../common/val_ids.csv`, seed 20261023). Runs on a laptop CPU; `transformer_scratch` uses a GPU if present.

## Setup

```bash
pip install -r requirements.txt          # Python 3.12
python ../common/make_split.py           # verifies the shared validation split
```

Kaggle CSVs go in `submissions/<team>/data/` (git-ignored), or pass `--data-dir PATH` / set `HOA_DATA_DIR`.

## Train / evaluate on the fixed split

```bash
python train.py --model <name>           # inner-CV search on the 8,000 rows, fit, score the 2,000
```

Writes `results/<name>/metrics.json` (chosen params, train/val AUROC, fit time, hardware), `val_pred.csv`,
`search.csv`, and the model's row in the team's testing history (`../SUBMISSIONS.md`).

## Kaggle submission

```bash
python train.py --model <name> --final   # refit on all 10,000 rows -> kaggle_submissions/<date>_<name>.csv
python predict.py --model <name> --out submission.csv   # re-create the file from artifacts/<name>/
```

## Models

| name | what |
|---|---|
| `template` | character-count logistic regression; the starting point for new models |
| `tfidf_lr` (C1) | word 1-2-gram + char_wb 2-5-gram TF-IDF -> logistic regression |

Adding a model: see [CONTRIBUTING.md](CONTRIBUTING.md).
