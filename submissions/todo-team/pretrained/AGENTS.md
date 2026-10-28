# AGENTS.md

Context for AI assistants (and humans) working in this repository: the problem,
how the code is laid out, how to run things, and what has been learned so far.
It is not a set of instructions about *how* to solve the problem — any approach
is welcome.

## The task

Kaggle competition **Cornell Tech CS5785 (Fall 2026) — AI vs Human text**.
Binary classification: predict the probability that a passage is AI-generated.

- Metric: **AUROC** on 3,000 test passages (public LB = 30%, private LB = 70%, final ranking uses private).
- Data (`data/`, not in git — download from Kaggle):
  `train.csv` (10,000 labeled, 50/50; `label` 1 = AI, 0 = human), `unlabeled.csv` (3,000),
  `test.csv` (3,000), `train_starter_2000.csv` (subset of train), `sample_submission.csv`.
- Passages are 75–350 words (≈210 avg) across many domains (recipes, abstracts, news, wiki bios, poems, reddit, …).
- The test set contains **writing styles, domains and generators not present in train**, so generalisation matters.
- Submission: `id,label` with exactly 3,000 rows, label in [0, 1]. Limit: 5 submissions/team/day. Deadline Nov 11, 2026.
- Course grade: 25% leaderboard; the rest is methodology — model comparison, effect of training-set size,
  use of unlabeled data, generalisation/error analysis, open-ended improvements, and a clean GitHub PR history.

## Machines

| | MacBook Pro (M1 Pro, 16 GB) | Desktop PC (Ryzen 5 9600X, 32 GB) |
|---|---|---|
| Accelerator | Apple GPU via MPS | RTX 5070 12 GB + RTX 3060 12 GB (CUDA, bf16) |
| Env | conda `cs-aml` | conda `cs-aml` with a CUDA 12.8+ PyTorch build (RTX 50xx needs it) |
| Profile | `python run_all.py --profile mac` | `python run_all.py --profile pc` |

- Torch device is one setting: `device:` in `configs/default.yaml` (`auto` | `cuda` | `cuda:<i>` | `mps` | `cpu`),
  `--device` on train/predict, or env var `AIVH_DEVICE`. Priority: `--device` / non-auto config > `AIVH_DEVICE` >
  auto-detect (cuda → mps → cpu). An unavailable device fails immediately. Saved models don't store the device.
- `python scripts/check_setup.py` verifies packages, data and GPUs (incl. whether the PyTorch build supports the card);
  `--models` also loads every pretrained model once.
- Pretrained weights download on first use into the Hugging Face cache; `scripts/download_models.py` fetches them up
  front (`--local` keeps a copy in `pretrained/`, which the code then prefers). `run_all.py` prefetches automatically.

## Directory layout

```
AGENTS.md / CLAUDE.md     this file (CLAUDE.md just imports it for Claude Code)
README.md                 setup + usage for humans
requirements.txt          pip deps; environment.yml creates the cs-aml conda env
train.py                  CV + test predictions + submission for ONE model config (entry point)
run_all.py                runs every job of an experiment profile in parallel (GPUs + CPU) and builds a report
predict.py                re-predict any CSV from a saved run
configs/
  default.yaml            shared settings: data, device, preprocessing, CV, semi-supervised, final strategy, outputs
  models/<config>.yaml    one file per model config (see table below); --model <config> selects it
  experiments/<profile>.yaml  job lists for run_all.py: smoke (does everything run?), pc, mac
src/
  config.py               loads default.yaml <- models/<config>.yaml <- --set key=value overrides
  data.py                 CSV loading, external data, CV splits (stratified / cluster / holdout), subsampling
  preprocessing.py        optional text-cleaning steps (each toggled in config)
  features.py             TfidfFeaturizer (word/char/subword) and stylometric features (base + extended)
  models/
    base.py               BaseModel interface: fit(texts, y, ...), predict_proba(texts), save/load
    __init__.py           REGISTRY mapping model class names -> classes
    tfidf_logreg.py  tfidf_nb.py  tfidf_svm.py      TF-IDF + linear models / NB / NB-SVM
    stylometric_gbm.py    hand-crafted style features -> LightGBM / random forest / hist GB
    lm_perplexity.py      causal-LM token statistics (+ Binoculars with an observer/performer pair) -> classifier
    embedding_logreg.py   frozen sentence-embedding model -> logistic regression
    transformer.py        fine-tuned pretrained encoder (bf16 on CUDA, gradient checkpointing, length-sorted inference)
  ensemble.py             rank averaging, greedy (Caruana) selection, logistic stacking on OOF predictions
  report.py               run_all report: summary table, sanity checks, ensembles, figures
  leaderboard.py          rewrites the Results table below from experiments/results.csv
  utils.py                seeding, metrics (AUROC, TPR@1%FPR, ...), results log + file lock, device resolver
scripts/
  check_setup.py          environment / GPU / data / model-loading check
  download_models.py      pre-download Hugging Face weights
  blend.py                blend chosen runs (rank / greedy / stack)
  record_kaggle_score.py  store a Kaggle public score for a run or submission
  add_external_data.py    clean/validate extra labeled data into data/external/
  generate_ai_text.py     generate extra AI passages with a local HF model
notebooks/01_eda.ipynb    exploratory data analysis
experiments/results.csv   one row per training run (tracked in git)
experiments/sessions/<session>/   run_all reports: report.md, summary.csv, ensembles.csv, sanity.csv, figures/ (tracked)
data/  outputs/  pretrained/      git-ignored: data, run artefacts/logs/submissions/caches, local model copies
```

## Model configs (`configs/models/`)

| config | what | hardware |
|---|---|---|
| `tfidf_logreg`, `tfidf_svm`, `tfidf_nb` | word+char TF-IDF → LR / linear SVM / NB-SVM | CPU |
| `stylometric_gbm` | 43 style/formatting features → LightGBM | CPU |
| `stylometric_gbm_ext` | + function words, POS-proxy rates, readability, coherence (Ma et al. 2023) | CPU |
| `lm_gpt2`, `lm_gpt2_medium` | GPT-2 token log-prob/rank/entropy stats + style → LightGBM | GPU or MPS |
| `binoculars_qwen05b`, `binoculars_qwen15b` | Binoculars (Qwen2.5 base + instruct) + token stats + style → LightGBM | MPS ok for 0.5B; 1.5B GPU |
| `binoculars_qwen15b_zeroshot` | raw Binoculars score, no training (shares the feature cache) | GPU |
| `embed_bge_base`, `embed_e5_base` | frozen sentence embeddings → LR | GPU or MPS |
| `ft_distilroberta`, `ft_roberta_base`, `ft_electra_base`, `ft_modernbert_base`, `ft_deberta_v3_small`, `ft_deberta_v3_base` | fine-tuned encoders | GPU; small ones possible on MPS |
| `ft_openai_detector`, `ft_openai_detector_large` | fine-tune OpenAI's RoBERTa GPT-2-output detector (as in Ma et al.) | GPU (large: 12 GB) |
| `ft_deberta_v3_large` | DeBERTa-v3-large, bf16 + gradient checkpointing | GPU 12 GB |

Feature-extractor models (LM, embeddings) cache features in `outputs/cache/` keyed by text hash.
Fine-tunes use `final.strategy: fold_average` (test prediction = mean of the CV fold models).

## How a single run works

```
python train.py --model <config> [--set key=value ...] [--device auto|cuda|mps|cpu] [--quick] [--cv-only] [--notes "..."]
```

1. Config = `configs/default.yaml` ← `configs/models/<config>.yaml` ← `--set` overrides (values parsed as YAML).
2. All splits are preprocessed with the same `preprocessing:` steps.
3. Cross-validation on `train.csv` per `cv.scheme`: `stratified` (k-fold), `cluster` (k-means topic clusters held
   out as whole groups — approximates unseen domains, lower scores, not comparable to stratified), `holdout`.
   External data and pseudo-labels only ever enter the training side of a fold; `data.train_frac` subsamples it.
4. Test predictions: `final.strategy` = `refit` (fit on all labeled data) or `fold_average`.
5. Outputs: `outputs/runs/<run_name>/` (config.yaml, oof.csv, metrics.json, test_pred.csv, unlabeled_pred.csv,
   status.json, model), `outputs/submissions/<run_name>.csv`; a row in `experiments/results.csv`; the table below.
6. Crash safety: each finished fold's predictions go to `outputs/runs/<run_name>/folds/` (keyed by a config + split
   fingerprint), and fine-tunes (`checkpoint_every_epoch: true`) save `outputs/checkpoints/<run_name>/fold_<k>/last.pt`
   after every epoch (model, optimizer, scheduler, scaler, RNG). Re-running with the same run name
   (`run_all --resume`, or `train.py --run-name <same>`) skips finished folds and continues the current fold after its
   last finished epoch. Checkpoints are deleted once a fold's predictions are saved (`output.keep_checkpoints=true`
   keeps them). `metrics.json` has `epoch_history` (train loss / fold-val AUROC per epoch, logged only — epochs stay
   fixed) and `epoch_val_auc_mean`.

Adding a model = a class in `src/models/` subclassing `BaseModel` + one line in `REGISTRY`, and/or a new
YAML in `configs/models/` (e.g. a different Hugging Face checkpoint for an existing class).

## Running everything (`run_all.py`)

```
python run_all.py --profile smoke    # every config on 2k texts / 2 folds / 1 epoch: catches setup errors
python run_all.py --profile pc       # full benchmark, both GPUs + CPU in parallel
python run_all.py --profile mac
python run_all.py --profile pc --only <job>,<job> | --skip <job> | --set cv.n_folds=3 | --gpus 0
python run_all.py --resume outputs/sessions/<session>     # continue after a crash / Ctrl+C
python run_all.py --report outputs/sessions/<session>     # rebuild the report
```

Each job is a `train.py` subprocess (log: `outputs/sessions/<session>/logs/<job>.log`), pinned to one GPU with
`CUDA_VISIBLE_DEVICES`; heaviest jobs start first; idle GPU workers also take CPU jobs; `after:` orders dependent jobs.
The report (`experiments/sessions/<session>/report.md`) has the results table, ensembles (rank mean, greedy
selection and LR stacking — the latter two cross-validated), learning-curve / semi-supervised / robustness
comparisons, per-topic AUROC, ROC curves (incl. low-FPR), score distributions, test-prediction correlations,
calibration, cost vs accuracy, and sanity checks (PASS/WARN/FAIL). The recommended upload is written to
`outputs/submissions/<session>__best.csv`.

## Observations so far

Data:
- Formatting differs systematically by class in train: blank lines (`\n\n`) appear in ~66% of AI passages and almost no
  human ones; human recipes often have fraction slashes stripped ("14 cup" for "1/4 cup"); double spaces are more common in
  human text. Test's blank-line rate (~32%) is consistent with the same pattern, but whether these cues hold for the unseen
  test domains/generators is unknown. The `preprocessing:` steps (and the `debiased_*` jobs) exist to test this —
  GPTZero (Adam et al. 2026) report that such formatting shortcuts in naive datasets hurt real-world robustness.
- **Topic shift:** with k-means (k=10) on TF-IDF over train+unlabeled+test, one cluster of creative writing
  (poems/stories: "love, life, world, day…") is ~41% of test but only ~6.5% of train. Unlabeled.csv matches train's topic
  mix, not test's. Adversarial validation (train vs test) AUROC ≈ 0.74.
- Some AI passages contain zero-width characters / Cyrillic homoglyphs, suggesting adversarial perturbations.
- AI-side cues: assistant openers ("Sure, here's…"), "Title:/Ingredients:" headers, heavy repetition (typical of small base
  models), all at different rates in test than in train → generator mix differs.

Models (baselines run on the Mac sandbox, default configs):
- Stylometric features + LightGBM are much stronger than TF-IDF n-grams (0.998 vs ~0.96 CV) and barely drop under
  `cluster` CV (0.994), whereas TF-IDF LR drops to ~0.91. Top features by gain: blank-line count, type-token ratio,
  burstiness, hapax ratio. Without any length/newline/spacing features stylometric GBM still reaches ~0.974.
- On the train passages in the creative-writing cluster, OOF AUROC is a bit lower for every model (TF-IDF LR 0.938, stylometric 0.993).
- Pseudo-labelling unlabeled.csv (1 round, 40% most confident) did not change TF-IDF LR `cluster` CV.
- Rank-blending TF-IDF LR + stylometric GBM (1:3) gives 0.998 OOF.
- The pretrained-model configs (LM features, Binoculars, embeddings, fine-tunes) were verified end-to-end only with
  tiny random stand-in models (Hugging Face was unreachable from the sandbox) — no real scores yet.

Related work notes (summaries from the team): GPTZero (Adam et al. 2026) — supervised detector, debiasing of formatting
cues, adversarial/paraphrase augmentation, reports recall at 1% FPR; weak on old LLMs (GPT-2). Ma et al. 2023 — syntax
features (POS, function words, punctuation, sentence length, perplexity) give F1 0.97 with LR; fine-tuned RoBERTa
detector relies on function words. Ponde et al. 2026 (theory) — guided paraphrasing can make AI text provably close to
human text; per-passage detection has a ceiling.

## Results

The table is regenerated automatically by `train.py`, `scripts/blend.py` and
`scripts/record_kaggle_score.py` from `experiments/results.csv`. Record Kaggle
scores with `python scripts/record_kaggle_score.py --run <run_name or submission file> --score <x>`.

<!-- LEADERBOARD:START -->
**Best Kaggle public score:** none recorded yet

| rank | run | config | CV scheme | CV AUROC | TPR@1%FPR | Kaggle public | train_frac | semi-sup | device | time (s) | author | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `20261028-111159_blend_lr_stylo` | blend_rank | stratified | 0.9981 |  |  | 1.0 | False |  |  | Claude | rank blend tfidf_logreg + stylometric_gbm (1:3) |
| 2 | `20261028-105015_stylometric_gbm` | stylometric_gbm | stratified | 0.9976 ± 0.0009 |  |  | 1.0 | False |  |  | Claude | initial baseline, default config |
| 3 | `20261028-105053_stylometric_gbm` | stylometric_gbm | cluster | 0.9937 ± 0.0025 |  |  | 1.0 | False |  |  | Claude | initial baseline, default config |
| 4 | `20261028-111158_blend_strat4` | blend_rank | stratified | 0.9851 |  |  | 1.0 | False |  |  | Claude | rank blend of LR, stylometric, SVM, NB (equal) |
| 5 | `20261028-105715_tfidf_svm` | tfidf_svm | stratified | 0.9591 ± 0.0046 |  |  | 1.0 | False |  |  | Claude | initial baseline, default config |
| 6 | `20261028-104136_tfidf_logreg` | tfidf_logreg | stratified | 0.9576 ± 0.0049 |  |  | 1.0 | False |  |  | Claude | initial baseline, default config |
| 7 | `20261028-105519_tfidf_nb` | tfidf_nb | stratified | 0.9463 ± 0.0054 |  |  | 1.0 | False |  |  | Claude | initial baseline, default config |
| 8 | `20261028-104555_tfidf_logreg` | tfidf_logreg | cluster | 0.9139 ± 0.0316 |  |  | 1.0 | False |  |  | Claude | initial baseline, default config |
| 9 | `20261028-110055_tfidf_logreg` | tfidf_logreg | cluster | 0.9128 ± 0.0326 |  |  | 1.0 | True |  |  | Claude | self-training on unlabeled (no gain vs 0.9139) |

_Top 12 of 9 logged runs by CV AUROC; full log in `experiments/results.csv`. CV scores from different schemes are not directly comparable (`cluster` CV holds out whole topics and is harder than `stratified`)._
<!-- LEADERBOARD:END -->

## Keeping this file current

When you change the directory structure, add a model/script/profile, change how
training or evaluation works, or learn something notable about the data or models,
update the relevant section above and add a line to the changelog below.
New best scores appear in the Results table automatically when runs are logged;
after a run_all session, add its headline findings to "Observations so far".

## Changelog

- 2026-10-28 — Per-epoch checkpointing + fold-level resume for long fine-tunes (see "How a single run works" step 6).
  Tested by killing a smoke `ft_deberta_v3_small` session mid-epoch (fold 2, epoch 2) and `--resume`-ing: fold 1 was
  skipped, fold 2 continued from epoch 1; fold AUROCs 0.9771/0.9811 vs 0.9760/0.9809 for an uninterrupted run
  (GPU nondeterminism only). Each fold is now seeded with `seed + fold`, so a fold's result doesn't depend on which
  earlier folds were skipped.

- 2026-10-28 — PC smoke run (23/23 jobs ok). Fixed: fine-tunes loaded DeBERTa-v3 in its stored fp16 dtype
  (transformers 5 default), so AdamW updated fp16 master weights and produced NaN after step 1 — weights are now
  always loaded in fp32 (autocast still runs bf16). Report shows the real exception for failed jobs; smoke profile
  no longer logs to `results.csv` (`log_results: false`); `run_all --resume` appends to job logs and accepts
  `--set` / `--only` / `--skip` for the re-run jobs; HF symlink warnings silenced on Windows.

- 2026-10-28 — Pretrained models: 9 fine-tune configs (DistilRoBERTa, RoBERTa, ELECTRA, ModernBERT, DeBERTa-v3
  small/base/large, OpenAI detector base/large), GPT-2 / GPT-2-medium perplexity features, Binoculars (Qwen2.5
  0.5B/1.5B pairs, incl. zero-shot), frozen BGE / E5 embeddings; extended stylometric features (Ma et al.).
  `run_all.py` multi-GPU scheduler with smoke/pc/mac profiles and an automatic report (tables, figures, sanity checks,
  ensembles); bf16 + gradient checkpointing; fold-averaged test predictions; TPR@1%FPR metric; results-log file lock;
  UTF-8 everywhere (Windows); `check_setup.py`, `download_models.py`, `CLAUDE.md`.
- 2026-10-28 — Configurable torch device (`device:` config key, `--device` on train/predict/generate, `AIVH_DEVICE` env var);
  fuller `.gitignore` (all of `data/` except README, model/cache files, credentials, editor files).
- 2026-10-28 — Initial baselines logged (TF-IDF LR/NB/SVM, stylometric GBM, blends); EDA findings added above.
- 2026-10-28 — Initial structure: EDA notebook, preprocessing pipeline, six models, train/predict entry points,
  CV schemes, pseudo-labelling, external-data and generation scripts, blending, auto-updated results table.
