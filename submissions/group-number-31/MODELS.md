# MODELS.md — what the six submitted models are and how they were built

Six models have a Kaggle file, one file each: three blends of pretrained models (track `pretrained/`, files A–C)
and three models trained from scratch (track `classical/`, files D–F). This page describes, for each one, what it
is made of, what was trained, and the exact pipeline and commands. Every number here comes from
`submission_log.csv` (shown in `SUBMISSIONS.md`) or from the results files named next to it.

- **Val AUROC**: model trained on 8,000 rows of `train.csv`, scored on the fixed 2,000 ids in `common/val_ids.csv`
  (seed 20261023). The same for all models.
- **Public AUROC**: Kaggle public leaderboard (30% of test).

| # | Submission | Track | Uploaded file | Val AUROC | Public AUROC |
|---|---|---|---|---|---|
| 1 | Blend A | pretrained | `pretrained/kaggle_submissions/2026-10-31_final_A_greedy_proxy.csv` | 0.99990 | **0.99161** |
| 2 | Blend B | pretrained | `pretrained/kaggle_submissions/2026-10-31_final_B_lm4.csv` | 0.99985 | 0.98848 |
| 3 | Blend C | pretrained | `pretrained/kaggle_submissions/2026-10-31_final_C_lm4_roberta.csv` | 0.99987 | 0.98845 |
| 4 | D: C2 `stylo_gbm` | classical | `classical/kaggle_submissions/final_D_stylo_gbm.csv` | 0.99831 | 0.95995 |
| 5 | E: C1 `tfidf_lr` (v2) | classical | `classical/kaggle_submissions/final_E_tfidf_lr.csv` | 0.99678 | 0.96374 |
| 6 | F: C3 `transformer_scratch` (v2) | classical | `classical/kaggle_submissions/final_F_transformer_scratch.csv` | 0.98903 | 0.91502 |

All six files are committed and validated: 3,000 rows each, ids match `sample_submission.csv`, values in [0, 1].
None of them needs retraining to be uploaded or reported.

E and F hold v2 of C1 and C3. Their v1 files, uploaded under the same names, scored public **0.88583** (C1) and
**0.83201** (C3) with val 0.96256 and 0.98500. They are kept in the log as `final_E_tfidf_lr@v1` and
`final_F_transformer_scratch@v1`, linked to the commits that hold them. Why v1 dropped and what v2 changed:
[Part 2, "Why C1 and C3 v1 fell on Kaggle"](#why-c1-and-c3-v1-fell-on-kaggle).

---

## Part 1 — The three pretrained blends

### 1.1 The big picture

```
                 ┌── member model 1 ──> P(AI) ──> percentile ──┐
passage text ────┼── member model 2 ──> P(AI) ──> percentile ──┼── weighted average ──> final score
                 └── member model k ──> P(AI) ──> percentile ──┘
```

Each blend is a **weighted average ensemble**:

- Every member model scores every passage.
- Each member's score is converted to a percentile, so the members are on the same 0–1 scale.
- The percentiles are averaged with fixed weights.

There is no gating (unlike a mixture of experts) and no voting.

There are two kinds of member, and they use the pretrained model in different ways:

| Kind | What the pretrained model does | Pretrained weights updated? | What makes the prediction |
|---|---|---|---|
| **A. Frozen-LM features** | Measures how predictable each token is (a feature extractor) | No, frozen | A LightGBM classifier trained on `train.csv` |
| **B. Fine-tuned encoder** | Is the classifier | Yes, all weights | The fine-tuned model itself |

### 1.2 Kind A: frozen language model → features → LightGBM

Code: `pretrained/src/models/lm_perplexity.py`. Configs: `pretrained/configs/models/lm_*.yaml`, `binoculars_*.yaml`.

**Idea.** A language model finds AI-generated text unusually predictable: its tokens have high probability, high
rank and little surprise. Human text is burstier. Measuring this does not depend on the topic.

**Pipeline, per passage:**

1. **Tokenize** with the LM's own tokenizer and truncate to 512 tokens. A BOS token is prepended so the first word
   is scored too. No other text cleaning is done (`preprocessing: strip` only), because formatting carries signal.
2. **Run the frozen LM** (one forward pass, no gradient). For every token, record:
   - its log-probability under the LM,
   - its rank among all vocabulary entries,
   - the entropy of the LM's predicted distribution.
3. **Summarise into 15 numbers:**
   - mean, standard deviation, minimum and 10th/25th/50th/75th/90th percentiles of the log-probabilities;
   - mean and standard deviation of the entropy;
   - surprisal divided by entropy;
   - share of tokens ranked 1st, and share ranked in the top 10;
   - mean log-rank;
   - number of tokens.
4. **Binoculars models only: 3 more numbers.** These use a second LM from the same family, the instruct version
   (the "performer"), next to the base model (the "observer"). The numbers are:
   - the performer's mean log-probability,
   - the cross-perplexity between the two models' next-token distributions,
   - the **Binoculars score** (Hans et al., 2024): performer log-perplexity ÷ cross-perplexity. It is low for
     machine text.
5. **Add 43 hand-crafted style features** (`pretrained/src/features.py`): lengths, sentence-length burstiness,
   type–token ratio, punctuation and function-word rates, blank lines, and so on. `lm_gpt2_nostyle` skips this step.
6. **Train LightGBM** on these 15–61 numbers with the `train.csv` labels. Settings: 500 trees, learning rate
   0.03, 15 leaves, subsample 0.8, column sample 0.8. This is the only trained part.

LM features are cached on disk by text hash (`outputs/cache/`), so after the first pass a LightGBM head retrains
in seconds.

| Member | Observer LM (performer) | Features | Fixed-split val AUROC |
|---|---|---|---|
| `lm_gpt2` | GPT-2, 124M | 15 LM + 43 style | 0.99972 |
| `lm_gpt2_nostyle` | GPT-2, 124M | 15 LM only | 0.98591 |
| `lm_gpt2_medium` | GPT-2-medium, 355M | 15 LM + 43 style | 0.99970 |
| `binoculars_qwen05b` | Qwen2.5-0.5B (Qwen2.5-0.5B-Instruct) | 18 LM + 43 style | 0.99978 |
| `binoculars_qwen15b` | Qwen2.5-1.5B (Qwen2.5-1.5B-Instruct) | 18 LM + 43 style | 0.99970 |

### 1.3 Kind B: fine-tuned encoder classifiers

Code: `pretrained/src/models/transformer.py`. Configs: `pretrained/configs/models/ft_*.yaml`.

**Pipeline:**

1. Load the pretrained encoder from Hugging Face with a 2-class classification head
   (`AutoModelForSequenceClassification`). RoBERTa-base and DeBERTa-v3 get a new, randomly initialised head. The
   OpenAI detector keeps its existing real/fake head, mapped onto our human/AI labels.
2. Tokenize with its own tokenizer, truncated to 512 tokens.
3. **Fine-tune all weights** on the `train.csv` labels:
   - AdamW with weight decay 0.01 and a linear schedule with 10% warmup;
   - bf16 on the GPU;
   - a fixed number of epochs, with no early stopping on validation data.
4. Output the softmax probability of the AI class.

| Member | Base checkpoint | Size | LR | Epochs | Batch × accum | Fixed-split val AUROC |
|---|---|---|---|---|---|---|
| `ft_roberta_base` | `FacebookAI/roberta-base` | 125M | 2e-5 | 3 | 16 × 2 | 0.99812 |
| `ft_deberta_v3_large` | `microsoft/deberta-v3-large` | 435M | 8e-6 | 2 | 4 × 8 | 0.99613 |
| `ft_openai_detector_large` | `openai-community/roberta-large-openai-detector` (RoBERTa-large already fine-tuned by OpenAI on GPT-2 output) | 355M | 8e-6 | 2 | 8 × 4 | 0.99836 |

### 1.4 How the uploaded blend files were produced

1. **Cross-validation runs.** `pretrained/run_all.py --profile ood` ran every member with 5-fold stratified CV on
   all 10,000 rows (seed 42, fold *k* seeded 42 + *k*). It used two GPUs (RTX 5070 + RTX 3060) and took about 4 h.
   Each run saved:
   - **out-of-fold predictions** for all 10,000 rows (every row scored by a model that never saw it);
   - **test predictions.** Kind A refits LightGBM on all 10,000 rows. Kind B averages its 5 fold models, with no
     extra refit.
2. **Percentile mapping** (`pretrained/src/ensemble.py: cdf_mean`). A member's test score becomes the fraction of
   that member's out-of-fold scores below it. The mapping is fixed by the training data, so it doesn't depend on
   what else is in test.
3. **Weighted average** with the weights below (`pretrained/make_submission.py`).

| Blend | Members (weight) | How the weights were chosen |
|---|---|---|
| **A** | `lm_gpt2_medium` ×3, `binoculars_qwen05b` ×3, `binoculars_qwen15b` ×2, `ft_deberta_v3_large` ×1, `ft_roberta_base` ×1, `lm_gpt2_nostyle` ×1, `ft_openai_detector_large` ×1 | Greedy forward selection with replacement (Caruana et al., 2004) on an out-of-distribution proxy, described below |
| **B** | `lm_gpt2`, `lm_gpt2_medium`, `binoculars_qwen05b`, `binoculars_qwen15b`, all ×1 | Equal weights: the four best LM-feature models by CV, chosen without the proxy |
| **C** | B's four + `ft_roberta_base`, all ×1 | Equal weights: tests whether adding one fine-tuned encoder helps |

**The out-of-distribution proxy (blend A only).** Test has creative writing and generators that `train.csv`
lacks, so we built a held-out set to choose weights. It was used **only to score models and pick weights, never
for training**, and the report discloses it (Q5.2 and Q6.3). It contains:

- 542 rewrites of training human passages, plus 816 poems and stories, generated locally by 7 open models never
  used for training: Llama-3.2-3B-Instruct, Qwen3-4B-Instruct, Phi-3.5-mini-instruct, SmolLM2-1.7B-Instruct,
  Granite-3.1-2B-Instruct, OLMo-2-1B-Instruct and Pythia-1.4B;
- 681 human poems and stories from held-out Project Gutenberg books.

Rules note: choosing 7 blend weights on labels we made ourselves is the one grey area in our work (the course says
supervised labels must come only from `train.csv`). Blend B and C involve no such choice.

**Reproducing.** These files are re-created exactly (max difference 0) from the saved runs, which are not in git:

```bash
python predict.py --model blend_a --out submission.csv            # also blend_b, blend_c; needs the saved runs + GPU
python predict.py --model blend_a --committed --out submission.csv # just copy the uploaded file
python train.py --model blend_a                                    # retrain the members from scratch (~4 h, 2 GPUs)
```

After a retrain, build the blend from the new run folders:

```bash
cd pretrained
python make_submission.py --run <run dirs> --weights 3 3 2 1 1 1 1 --name blend_a
```

The record: `pretrained/results/reproduction.csv`, and the research log in `pretrained/experiments/results.csv`
(72 runs).

---

## Part 2 — The three from-scratch (classical) models

No pretrained weights, embeddings or tokenizers. Everything is learned from `train.csv` (plus `unlabeled.csv`
text for C3's tokenizer).

**Shared pipeline** (`classical/train.py`):

1. Load `train.csv`. Split it into the 8,000 training rows and the fixed 2,000 validation rows.
2. **Hyperparameter search**, if the config has a `search:` grid: 5-fold CV inside the 8,000 rows; the best mean
   AUROC wins. The 2,000 validation rows are never used to choose anything.
3. Fit on the 8,000 rows with the chosen settings, then score the 2,000 rows. This writes
   `classical/results/<model>/metrics.json`, `val_pred.csv` and `search.csv`.
4. `--final`: refit on all 10,000 rows and predict `test.csv`. This writes
   `classical/kaggle_submissions/final_<X>_<model>.csv`, validates it, and saves the model to `classical/artifacts/`
   (git-ignored). Each model keeps its letter (D, E, F); a refit replaces its file and resets the log row to
   "pending upload".

All seeds are 20261023.

### C1 `tfidf_lr` — TF-IDF + logistic regression

Code: `classical/models/tfidf_lr.py`. Config: `classical/configs/tfidf_lr.yaml`.

1. Three TF-IDF views, each with sublinear term frequency and up to 300k features, vocabulary and IDF fit on the
   training rows only:
   - word 1–2-grams (min_df 2);
   - character 1–5-grams over the **raw** text (min_df 3, case kept), so line breaks, blank lines and spacing
     are features;
   - word 1–3-grams over **text-distorted** input (`classical/features/distortion.py`, Stamatatos 2017): every
     word outside the 300 most frequent training words becomes `*` (or `#` if capitalised), and line breaks are
     tokens. This keeps function-word, punctuation and line patterns and drops topic.
2. The three sparse matrices are stacked side by side.
3. L2 logistic regression (liblinear).

Search: C ∈ {4, 16, 64} × lowercase ∈ {yes, no}. Chosen: **C = 64, lowercase** (inner CV 0.9956; the grid is
flat within 0.0005). Fit time 23 s on CPU. Val AUROC 0.99678.

v1 (the first `final_E` upload, public 0.88583) had only word 1–2-grams and character 2–5-grams *inside* word
boundaries (`char_wb`), val 0.96256.

### C2 `stylo_gbm` — stylometric features + LightGBM (best classical model)

Code: `classical/models/stylo_gbm.py`, features in `classical/features/stylometry.py`. Config:
`classical/configs/stylo_gbm.yaml`.

1. Turn each passage into **101 hand-crafted numbers**:
   - length, sentence-length mean, standard deviation and burstiness;
   - type–token and hapax ratios, repeated n-grams;
   - punctuation rates, function-word rates, readability, adjacent-sentence coherence;
   - **layout**: blank lines, words per line, share of short lines;
   - **character classes**: zero-width characters, Cyrillic/Greek homoglyphs, curly quotes, dashes.
2. LightGBM on those numbers.

Search: leaves ∈ {15, 31, 63} × learning rate ∈ {0.03, 0.1} × trees ∈ {300, 800}. Chosen: **63 leaves, learning
rate 0.03, 800 trees**. Fit time 7 s on CPU. Val AUROC 0.99831.

What we learned: layout features carry 50.4% of the model's split gain, blank lines alone 37.5%
(`classical/results/stylo_gbm/importances.csv`). Without them, val AUROC falls to 0.98724
(`classical/results/stylo_gbm@no_layout/`).

### C3 `transformer_scratch` — small Transformer, random initialisation

Code: `classical/models/transformer_scratch.py`. Config: `classical/configs/transformer_scratch.yaml`.

1. **Input.** Single line breaks are joined into spaces (blank lines, i.e. paragraph and stanza breaks, stay),
   and the text is distorted like C1's third view (words outside the 300 most frequent training words masked).
2. **Tokenizer trained here.** A byte-level BPE with a 12k vocabulary, trained on that input for the training
   rows plus `unlabeled.csv`, and never on validation or test text. Byte-level keeps case, spaces and blank lines.
3. **Model.** Learned token and position embeddings (max 512 tokens) feed a **4-layer pre-norm Transformer
   encoder** (d_model 256, 4 heads, feed-forward 1024, dropout 0.1). Token outputs are mean-pooled into a linear
   head. All weights start random.
4. **Training.**
   - AdamW: learning rate 3e-4, weight decay 0.01, 6% warmup then linear decay, batch 32, bf16 on GPU.
   - Up to 20 epochs, with early stopping (patience 3) on a stratified 10% inner split of the *training* rows;
     the best epoch is restored.

No grid search: the architecture was fixed up front because a 5-fold grid would take hours. Fit time 32 s on the
RTX 5070. Val AUROC 0.98903.

v1 (the first `final_F` upload, public 0.83201) read the raw text without line joining or distortion, val 0.98500.

### Why C1 and C3 v1 fell on Kaggle

v1 of C1 and C3 looked fine on validation (0.963 and 0.985) but scored 0.886 and 0.832 on the public
leaderboard. The fixed validation split comes from train's distribution, so it cannot show the test shift:
unseen generators and about 17% verse, against 1.6% of train (`common/results/split_profile.csv`).

`classical/ood_eval.py` measures this without touching Kaggle. It fits on the 8,000 training rows and scores
held-out text: AI rewrites by 7 generators not in train (only rewrites of validation passages), and human vs AI
poems/stories with the human side from held-out Project Gutenberg books. These are the pretrained track's
evaluation sets, never trained on. Everything is pooled into one `proxy_auc`, mixed like test. Results:
`classical/results/ood_eval.csv`.

| Model | Val | Unseen generators | Creative | Human verse flagged* | Proxy | Public |
|---|---|---|---|---|---|---|
| C1 v1 (`final_E` v1) | 0.9626 | 0.8728 | 0.7801 | 7% | 0.8708 | 0.88583 |
| C1 v2 (`final_E`) | 0.9968 | 0.9538 | 0.9237 | 31% | 0.9555 | 0.96374 |
| C3 v1 (`final_F` v1) | 0.9817 | 0.8791 | 0.6105 | 71% | 0.7699 | 0.83201 |
| C3 v2 (`final_F`) | 0.9881 | 0.8694 | 0.9329 | 24% | 0.9258 | 0.91502 |
| C2 (`final_D`, unchanged) | 0.9983 | 0.9665 | 0.9343 | 78% | 0.9650 | 0.95995 |

\* share of held-out Gutenberg human poems/stories scored above the threshold that flags 5% of validation humans.

- The proxy put the three v1 models in the same order as Kaggle, 0.01–0.06 below their public scores.
- **C1 v1** read topic: word n-grams and in-word character n-grams never see layout, and topic does not carry
  over to unseen domains. Raw character n-grams and the distorted-text view fixed most of it.
- **C3 v1** learned "short lines = AI": in train, human passages have long lines and AI ones short lines, so it
  scored most human verse as AI. Joining single line breaks removed that shortcut (creative AUROC 0.61 → 0.93).
  Distortion helped on top. Cropping to random windows and a smaller model did not (variants in `ood_eval.csv`).
- The C3 v2 design gave proxy 0.916–0.924 over three seeds.
- The proxy was used to compare designs only. Hyperparameters are still chosen by inner CV, never on validation
  or Kaggle.

### Commands

```bash
python train.py --model stylo_gbm                              # validate on the fixed split
python train.py --model stylo_gbm --final                      # refit on 10k, write final_D_stylo_gbm.csv + artifacts
python predict.py --model stylo_gbm --out submission.csv       # re-create it from artifacts/
python predict.py --model stylo_gbm --committed --out submission.csv   # copy the committed file
python ood_eval.py --model stylo_gbm                           # score on the held-out shifted text
```

The same commands work for `tfidf_lr` (file E) and `transformer_scratch` (file F). A fresh clone
reproduced C2's committed file exactly, and `predict.py` reproduces E and F exactly from the saved models.

### Other classical models (validated, not submitted)

These are in `classical/results/` and appear in the report as comparisons:

- `linear_svm` 0.9623
- `nbsvm` 0.9699
- `rf_stylo` 0.9969
- `ensemble_c123` 0.99898: a percentile blend of C1–C3 (v2 C1 and C3). Its weights come from greedy selection on out-of-fold
  predictions inside the 8,000 training rows, and they ended up 0.65 C2 / 0.30 C1 / 0.05 C3 (`results/ensemble_c123/metrics.json`).
  With v1 C1 and C3 they were 0.9 / 0.1 / 0 and val 0.99868.
