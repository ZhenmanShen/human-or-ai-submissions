# team_pls_give_us_an_A — Human or AI?

Code for the CS5785 "Human or AI?" Kaggle competition. Six models were submitted, three from each of two
tracks, and every model is scored on the same fixed 2,000-row validation split.

## Start here

| I want to... | Go to |
|---|---|
| Read the report | [`reports/kaggle_report.tex`](reports/kaggle_report.tex) (the one report we submit) |
| See every Kaggle upload and its score | [`SUBMISSIONS.md`](SUBMISSIONS.md) |
| See the six models side by side | [`common/results/compare_tracks.csv`](common/results/compare_tracks.csv), [`common/figures/six_models.pdf`](common/figures/six_models.pdf) |
| Run a classical model on a laptop | [`classical/README.md`](classical/README.md) |
| Add a new model | [`classical/CONTRIBUTING.md`](classical/CONTRIBUTING.md) |
| Run the pretrained (GPU) pipeline | [`pretrained/README.md`](pretrained/README.md) |
| See what is still to do | search the repo for `[TODO` |
| Brief an AI assistant (Claude Code, Codex, ...) | [`AGENTS.md`](AGENTS.md): task, layout, models, rules, TODOs in one page |

## Folder map

```
submissions/team_pls_give_us_an_a/
├── README.md              this file: setup, commands, seeds, AI-use disclosure
├── AGENTS.md              one-page briefing for AI assistants and new teammates (CLAUDE.md loads it)
├── SUBMISSIONS.md         every Kaggle upload and its score (generated from submission_log.csv)
├── submission_log.csv
├── reports/               THE REPORT: kaggle_report.tex + build_tables.py (picks rows from the track tables)
├── common/                shared by both tracks
│   ├── val_ids.csv          the fixed validation split (2,000 ids, seed 20261023)
│   ├── data.py metrics.py   data loading, AUROC, length/verse subgroups
│   ├── compare_tracks.py    the six-model comparison -> results/ tables/ figures/
│   └── update_submission_log.py, validate_submission.py, make_split.py, describe_splits.py
├── classical/             from-scratch models, no pretrained weights (CPU-friendly)
│   ├── models/              one file per model: tfidf_lr (C1), stylo_gbm (C2), transformer_scratch (C3), + extras
│   ├── configs/             one YAML per model, including its hyperparameter grid
│   ├── train.py predict.py experiments.py report_tables.py
│   ├── results/             metrics + predictions per model (all report numbers come from here)
│   ├── kaggle_submissions/  the uploaded CSVs
│   └── reports/             tables/figures used by the report (+ a longer classical-only draft)
└── pretrained/            frozen-LM features + fine-tuned encoders, blended (needs a GPU)
    ├── src/ configs/        the pipeline, imported with its history from the research repo
    ├── train.py run_all.py make_submission.py scripts/
    ├── results/             fixed-split metrics, curves, error analysis (all report numbers come from here)
    ├── kaggle_submissions/  the uploaded CSVs (blends A, B, C)
    ├── experiments/         log of every research run (72 runs, 5-fold CV, seed 42)
    └── reports/             tables/figures used by the report (+ a longer pretrained-only draft)
```

The six submitted models:

| Model | Track | Code |
|---|---|---|
| Blend A (best public score, 0.99161) | pretrained | `pretrained/make_submission.py`, weights 3 3 2 1 1 1 1 |
| Blend B, Blend C | pretrained | same script, equal weights |
| C1 TF-IDF + logistic regression | classical | [`classical/models/tfidf_lr.py`](classical/models/tfidf_lr.py) |
| C2 stylometric LightGBM (best classical) | classical | [`classical/models/stylo_gbm.py`](classical/models/stylo_gbm.py) |
| C3 small Transformer from scratch | classical | [`classical/models/transformer_scratch.py`](classical/models/transformer_scratch.py) |

Notes:

- **The report.** Its tables are generated: run the track scripts, then `python common/compare_tracks.py` and
  `python reports/build_tables.py`.
  - To compile it in Overleaf, upload the whole team folder; the report reads `../common`, `../pretrained` and `../classical`.
  - `pretrained/reports/kaggle_pretrained_report.tex` and `classical/reports/kaggle_classical_report.tex` are
    longer single-track drafts that cover every model. They are background only and not submitted.
- **AI-assistant files.**
  - [`AGENTS.md`](AGENTS.md) in this folder is the team-wide briefing.
  - `pretrained/AGENTS.md` is the research repo's log of every experiment, carried over with the pretrained track.
    `pretrained/scripts/` still update its results table, so leave it where it is.
- **Why the deep path.** The course only accepts PRs that touch `submissions/<team-name>/`, so all our files
  live in this folder, named after our Kaggle team (lowercase, as the course asks).
- **Branches.**
  - `main` holds everything; work there, or on a short-lived branch merged into `main`.
  - `submission/team_pls_give_us_an_a` is the branch the course PR will be opened from. It equals `main` minus the two
    pointer files at the repo root (`AGENTS.md`, `CLAUDE.md`), which point AI assistants here and must stay out of
    the PR. `AGENTS.md` shows how to update it.
  - `pretrained`, `classical` and `model/*` are older per-track and per-model branches. They are already merged into
    `main` and kept only for the commit history.

## 1. Team

- Kaggle team name: `team_pls_give_us_an_A`
- Members (name, NetID): `[TODO]`

## 2. Environment

Python 3.12. Use one environment per track; the classical one is a subset:

```bash
pip install -r classical/requirements.txt     # numpy, pandas, scikit-learn, lightgbm, pyyaml, matplotlib, torch, tokenizers
pip install -r pretrained/requirements.txt    # adds transformers etc.
```

An RTX 50-series GPU needs a CUDA 12.8+ PyTorch build; see `pretrained/README.md`. The classical track runs on a laptop CPU.

Competition data is never committed. Put `train.csv`, `test.csv`, `unlabeled.csv` and `sample_submission.csv` in
`submissions/<team>/data/` (git-ignored), or point to them with `--data-dir PATH` or `HOA_DATA_DIR`.

## 3. Validation protocol and train/evaluate commands

```bash
cd submissions/<team>
python common/make_split.py          # must print "OK: val_ids.csv reproduced exactly"
```

`common/val_ids.csv` holds the ids and labels of a fixed stratified 2,000-row validation set from `train.csv`:
1,000 human and 1,000 AI, made with `train_test_split` and seed `20261023`.

- Every model in both tracks trains on the other 8,000 rows and is scored on these 2,000.
- Hyperparameters are chosen by 5-fold CV inside the 8,000 rows, never on the validation set or on Kaggle.
- Final Kaggle models are refit on all 10,000 rows.

Classical (from `classical/`):

```bash
python train.py --model stylo_gbm            # also tfidf_lr, transformer_scratch, linear_svm, nbsvm, rf_stylo, ensemble_c123
python experiments.py curve | semisup | errors
python report_tables.py
```

Pretrained (from `pretrained/`):

```bash
python run_all.py --profile fixed
python scripts/report_experiments.py collect --session outputs/sessions/<session>   # then: hpsearch, curve, semisup, errors, debias
python scripts/report_tables.py
```

Both tracks (from the team folder):

```bash
python common/compare_tracks.py              # six-model table + figures
python common/describe_splits.py             # line-structure profile of each split
```

## 4. Command that writes `submission.csv`

Final submission: `[TODO: team decision]`. Best public score so far: pretrained blend A, 0.99161. Every uploaded file is
committed, and its row in `SUBMISSIONS.md` links the commit that added it.

- **Pretrained blend A**:

  ```bash
  python run_all.py --profile ood
  ```

  This trains the 5-fold models (seed 42, ~4 h on two 12 GB GPUs). Then:

  ```bash
  python make_submission.py --run <7 runs> --weights 3 3 2 1 1 1 1 --name final_A_greedy_proxy
  ```

  This writes `pretrained/outputs/submissions/final_A_greedy_proxy.csv`, which matches
  `pretrained/kaggle_submissions/2026-10-31_final_A_greedy_proxy.csv` exactly when run on the saved models
  (`pretrained/results/reproduction.csv`).

- **Classical C2** (laptop CPU, ~10 s):

  ```bash
  python train.py --model stylo_gbm --final
  python predict.py --model stylo_gbm --out submission.csv
  ```

  The first command writes `classical/kaggle_submissions/<date>_stylo_gbm.csv`. The second re-creates the same file
  from `artifacts/`.

## 5. Seeds and important config

- Split, CV folds and every new experiment: `20261023`.
- The pretrained research runs behind the uploaded files used seed `42`, with fold *k* seeded `42 + k`.
- Fine-tunes and `transformer_scratch` train in bf16 on GPU, so reruns can differ in the last digits.
- No external data is used for training. The pretrained track's out-of-distribution proxy (text from seven local
  generators and Project Gutenberg books) was used only to evaluate models and choose the blend.

## 6. AI-tool disclosure

[TODO: team to edit; must match Q6.3 in the report.]

Claude Code (Anthropic) wrote most of the code in both tracks, ran the experiments on the team's machines and
drafted both reports. Every reported number is generated from committed results files by scripts in this folder.
[TODO: what the team verified or changed.]

Pretrained weights are used only in `pretrained/`: GPT-2 and GPT-2-medium, Qwen2.5-0.5B/1.5B (base and instruct),
RoBERTa-base, DeBERTa-v3-large and the OpenAI RoBERTa-large GPT-2-output detector. `classical/` uses none.
No external LLM API labelled or scored test passages, and no hidden labels were retrieved.
