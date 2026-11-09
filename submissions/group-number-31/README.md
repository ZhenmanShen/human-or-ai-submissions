# Group Number 31 — Human or AI?

Code for the CS5785 "Human or AI?" Kaggle competition. Six models were submitted, three from each of two
tracks, and every model is scored on the same fixed 2,000-row validation split.

## Start here

| I want to... | Go to |
|---|---|
| Understand how each model is built | [`MODELS.md`](MODELS.md): pipelines of all six submitted models |
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
submissions/group-number-31/
├── README.md              this file: setup, commands, seeds, AI-use disclosure
├── MODELS.md              how each of the six submitted models is built
├── AGENTS.md              one-page briefing for AI assistants and new teammates (CLAUDE.md loads it)
├── train.py predict.py    team entry points (like the course's _example_team); they call the track scripts
├── requirements.txt       both tracks (classical/requirements.txt alone for a laptop)
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
│   ├── train.py predict.py experiments.py report_tables.py ood_eval.py
│   ├── results/             metrics + predictions per model (all report numbers come from here)
│   ├── kaggle_submissions/  one CSV per model: final_D_stylo_gbm, final_E_tfidf_lr, final_F_transformer_scratch
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

| Model (`--model` name) | Track | Code |
|---|---|---|
| Blend A `blend_a` (best public score, 0.99161) | pretrained | `pretrained/make_submission.py`, weights 3 3 2 1 1 1 1 |
| Blend B `blend_b`, Blend C `blend_c` | pretrained | same script, equal weights |
| D: C2 stylometric LightGBM `stylo_gbm` (public 0.95995) | classical | [`classical/models/stylo_gbm.py`](classical/models/stylo_gbm.py) |
| E: C1 TF-IDF + logistic regression `tfidf_lr` (best classical public score, 0.96374) | classical | [`classical/models/tfidf_lr.py`](classical/models/tfidf_lr.py) |
| F: C3 small Transformer from scratch `transformer_scratch` | classical | [`classical/models/transformer_scratch.py`](classical/models/transformer_scratch.py) |

See [`MODELS.md`](MODELS.md) for each model's pipeline. E and F were rebuilt (v2) after their first uploads scored
0.88583 and 0.83201; v2 scored 0.96374 and 0.91502. MODELS.md explains why v1 fell on Kaggle and what changed.

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
  live in this folder, named after our Kaggle team "Group Number 31" (lowercase, hyphens for spaces, as the
  course asks).
- **`_example_team/`** next to this folder belongs to the course (its README says not to modify it), so it stays.
  Our layout follows it: `README.md`, `requirements.txt`, `train.py` and `predict.py` at the top of the team folder.
- **Branches.** There is one branch, `main`. The course PR has to come from a branch named
  `submission/group-number-31`; `AGENTS.md` shows how to create it from `main` when we open the PR.

## 1. Team

- Kaggle team name: `Group Number 31` (folder and PR branch: `group-number-31`)
- Members (name, NetID): `[TODO]`

## 2. Environment

Python 3.12.

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt                       # both tracks
pip install -r classical/requirements.txt             # or: classical track only (laptop CPU is enough)
```

The pretrained track needs a 12 GB NVIDIA GPU. An RTX 50-series card needs a CUDA 12.8+ PyTorch build, installed
before the requirements; see `pretrained/README.md`.

Competition data is never committed. Put `train.csv`, `test.csv`, `unlabeled.csv` and `sample_submission.csv` in
`data/` inside this folder (git-ignored), or pass `--data-dir PATH`, or set `HOA_DATA_DIR`.

## 3. Validation protocol and train/evaluate commands

All commands run from this folder.

```bash
python common/make_split.py          # must print "OK: val_ids.csv reproduced exactly"
```

`common/val_ids.csv` holds the ids and labels of a fixed stratified 2,000-row validation set from `train.csv`:
1,000 human and 1,000 AI, made with `train_test_split` and seed `20261023`.

- Every model in both tracks trains on the other 8,000 rows and is scored on these 2,000.
- Hyperparameters are chosen by 5-fold CV inside the 8,000 rows, never on the validation set or on Kaggle.
- Final Kaggle models are refit on all 10,000 rows.

```bash
python train.py --model stylo_gbm --seed 20261023    # train on 8k, score the fixed 2k -> classical/results/stylo_gbm/
                                                     # also tfidf_lr, transformer_scratch, linear_svm, nbsvm, rf_stylo, ensemble_c123
python train.py --model blend_b                      # pretrained: retrain a blend's members, 5-fold CV (GPU, hours)
```

Report experiments:

```bash
cd classical  && python experiments.py curve | semisup | errors && python report_tables.py && cd ..
cd pretrained && python run_all.py --profile fixed && python scripts/report_experiments.py collect --session outputs/sessions/<session>
              # then: hpsearch, curve, semisup, errors, debias; then python scripts/report_tables.py
python common/compare_tracks.py && python reports/build_tables.py    # six-model table, figures, report tables
```

## 4. Command that writes `submission.csv`

Final submission: `[TODO: team decision]`. Best public score so far: pretrained blend A, 0.99161. Every uploaded
file is committed (`*/kaggle_submissions/`), and its row in `SUBMISSIONS.md` links the commit that added it.

```bash
python predict.py --model blend_a --committed --out submission.csv   # the exact uploaded file, no compute
python train.py --model stylo_gbm --final                            # classical: refit on 10k (~10 s CPU)
                                                                     #   -> classical/kaggle_submissions/final_D_stylo_gbm.csv
python predict.py --model stylo_gbm --out submission.csv             #   then predict from the saved model
python predict.py --model blend_a --out submission.csv               # pretrained: re-predict from the saved runs
```

- Pretrained blends re-predict test with the saved 5-fold runs, which are too large for git (0.3–1.7 GB per
  fold). They live in the research repo's `outputs/runs/`, or pass `--runs-dir`.
- Run on those runs, this reproduces all three uploaded files exactly (`pretrained/results/reproduction.csv`).
- Retraining them from scratch takes about 4 h on two 12 GB GPUs; see `MODELS.md`.

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
