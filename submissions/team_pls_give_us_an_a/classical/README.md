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
python train.py --model <name> --final --letter <X>   # refit on all 10,000 rows
                                                       # -> kaggle_submissions/final_<X>_<name>.csv
python predict.py --model <name> --out submission.csv   # re-create the file from artifacts/<name>/
```

Each upload gets the next free letter after the pretrained track's A-C: D = `stylo_gbm`, E = `tfidf_lr` v1,
F = `transformer_scratch` v1, G = `tfidf_lr` v2, H = `transformer_scratch` v2 (scores in `../SUBMISSIONS.md`).

## Checking a model against the test shift

```bash
python ood_eval.py --model <name> [--set KEY=VALUE ...] [--tag <variant>]
```

The fixed validation split comes from train's distribution, so it cannot show the drop to Kaggle (test has unseen
generators and ~17% verse). `ood_eval.py` fits on the 8,000 training rows and scores held-out text from 7 unseen
generators and Project Gutenberg verse/stories, pooled into one `proxy_auc` (`results/ood_eval.csv`). It tracked
the v1 public scores (proxy 0.965 / 0.871 / 0.770 for C2 / C1 / C3; public 0.960 / 0.886 / 0.832). Use it to
compare designs; hyperparameters are still chosen by inner CV only.

## Models

| name | what |
|---|---|
| `template` | character-count logistic regression; the starting point for new models |
| `tfidf_lr` (C1) | word 1-2-gram + raw char 1-5-gram + word 1-3-grams over text-distorted input (features/distortion.py) TF-IDF -> logistic regression |
| `stylo_gbm` (C2) | 101 hand-crafted stylometric features (features/stylometry.py) -> LightGBM |
| `transformer_scratch` (C3) | 4-layer Transformer encoder (d=256) from random init over a 12k byte-level BPE trained on train + unlabeled text; input has single line breaks joined and is text-distorted |
| `linear_svm` | C1 v1's TF-IDF features (word + char_wb) -> linear SVM (comparison row) |
| `nbsvm` | NB-SVM: NB log-count-ratio scaled binary n-grams -> LR (comparison row) |
| `rf_stylo` | C2's stylometric features -> random forest (comparison row) |
| `ensemble_c123` | C1+C2+C3 rank/CDF blend, weights by greedy selection on inner-CV OOF predictions |

Adding a model: see [CONTRIBUTING.md](CONTRIBUTING.md).
