# [TODO: Kaggle team name] — Human or AI?

Code for the CS5785 "Human or AI?" Kaggle competition. There are two tracks, and both use the same validation split:

| Folder | What | Report |
|---|---|---|
| [`pretrained/`](pretrained/) | Frozen-LM features (GPT-2, Qwen2.5 Binoculars) and fine-tuned encoders (RoBERTa, DeBERTa-v3, OpenAI detector), blended. Imported with its development history. | [`pretrained/reports/kaggle_pretrained_report.tex`](pretrained/reports/kaggle_pretrained_report.tex) |
| [`classical/`](classical/) | Models trained from scratch, no pretrained weights: C1 TF-IDF + LR, C2 stylometric LightGBM, C3 small Transformer, plus comparison rows | [`classical/reports/kaggle_classical_report.tex`](classical/reports/kaggle_classical_report.tex) |
| [`common/`](common/) | The fixed validation split, data loading, metrics, the submission validator, the submission log and the cross-track comparison | — |

- **The report:** [`reports/kaggle_report.tex`](reports/kaggle_report.tex). It covers both tracks and centres on the six models submitted to Kaggle, three per track.
  - Its tables are assembled by `reports/build_tables.py` from the tracks' generated tables.
  - The per-track reports above are longer versions with every model.
  - To compile it in Overleaf, upload the whole team folder; the report reads `../common`, `../pretrained` and `../classical`.
- Submission and testing history: [`SUBMISSIONS.md`](SUBMISSIONS.md), generated from `submission_log.csv`.
- Adding a model: [`classical/CONTRIBUTING.md`](classical/CONTRIBUTING.md).

## 1. Team

- Kaggle team name: `[TODO]`
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
