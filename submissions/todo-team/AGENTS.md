# AGENTS.md — team folder for the CS5785 "Human or AI?" Kaggle project

Read this first. It is the context file for AI coding assistants (Claude Code, Codex, Cursor, ...) and for
teammates. Human-facing setup and commands are in [README.md](README.md).

## The task

- Kaggle competition `cornell-tech-cs-5785-combined-2026-ai-vs-human-text`: predict P(AI-written) for 3,000
  test passages. Metric: AUROC. Public leaderboard = 30% of test, private = 70%. 5 uploads per day per team.
- Data (never committed): `train.csv` (10,000 labelled, 50/50, `label` 1 = AI), `unlabeled.csv` (3,000),
  `test.csv` (3,000), `sample_submission.csv`. Test contains unseen domains and generators: about 17% of test is
  verse against 1.6% of train (`common/results/split_profile.csv`).
- Grading: a LaTeX report from the course template (`reports/kaggle_report.tex`), the private Kaggle score, and
  the code plus its commit history, all submitted as **one pull request** from this fork to
  `sekhari/human-or-ai-submissions`.

## Layout

```
human-or-ai-submissions/            the team's fork of the course repo
├── README.md, STAFF.md, .github/   course files: never edit
└── submissions/
    ├── _example_team/              course example: never edit
    └── todo-team/                  EVERYTHING of ours (rename to the Kaggle team name, see TODOs)
        ├── AGENTS.md               this file
        ├── README.md               setup, commands, seeds, AI-use disclosure (required by the course)
        ├── SUBMISSIONS.md          every model and Kaggle upload; generated from submission_log.csv
        ├── reports/                kaggle_report.tex, the one report we submit; build_tables.py
        ├── common/                 shared by both tracks: val split, data loading, metrics, log, comparison
        ├── classical/              from-scratch models, no pretrained weights, CPU-friendly
        └── pretrained/             pretrained-LM pipeline imported with history from the research repo
                                    (has its own AGENTS.md: the research log of 72 runs)
```

Why the deep path: the course README only allows a team's PR to touch `submissions/<team-name>/`.
Don't move our files out of the team folder.

## Two tracks, one validation protocol

All numbers in the report are measured the same way. The 2,000 ids in `common/val_ids.csv` are a stratified
split with seed 20261023. Models train on the other 8,000 rows and are scored on those 2,000. Hyperparameters are
picked by 5-fold CV inside the 8,000 rows, never on the validation rows. Final Kaggle models are refit on all
10,000 rows. `python common/make_split.py` checks that the split reproduces exactly.

| Track | Submitted models | Val AUROC | Public AUROC |
|---|---|---|---|
| pretrained | Blend A (7 members, weights 3 3 2 1 1 1 1) | 0.99990 | 0.99161 |
| pretrained | Blend B (4 LM-feature models, equal weights) | 0.99985 | 0.98848 |
| pretrained | Blend C (B + RoBERTa-base, equal weights) | 0.99987 | 0.98845 |
| classical | C1 `tfidf_lr`: word + char TF-IDF → logistic regression | 0.96256 | pending upload |
| classical | C2 `stylo_gbm`: 101 stylometric features → LightGBM | 0.99831 | pending upload |
| classical | C3 `transformer_scratch`: BPE + 4-layer Transformer, random init | 0.98500 | pending upload |

Source of these numbers: `submission_log.csv` / `SUBMISSIONS.md`. Always re-read them there; this table is a
snapshot.

### pretrained/ — how the blends work

There are two kinds of member models.

1. **Frozen LM as a feature extractor, LightGBM as the classifier**
   (`src/models/lm_perplexity.py`, configs `lm_gpt2*`, `binoculars_qwen*`).
   - GPT-2, GPT-2-medium or Qwen2.5 (base + instruct) score each token of the text; their weights are never updated.
   - The per-token log-probability, rank and entropy are summarised into 15 numbers per text. Binoculars models
     add 3 more: the performer's mean log-probability, the observer–performer cross-perplexity and the Binoculars score.
   - Most of these models also add 43 hand-written style features.
   - A LightGBM model trained on these features outputs P(AI).
2. **Fine-tuned encoder classifiers** (`src/models/transformer.py`, configs `ft_*`).
   - RoBERTa-base, DeBERTa-v3-large and the RoBERTa-large OpenAI detector, fine-tuned end to end with a
     classification head.
   - Test predictions average the 5 cross-validation fold models.

A blend maps each member's test score to a percentile using that member's out-of-fold scores, then takes a
weighted average (`src/ensemble.py: cdf_mean`). Every member scores every text: there is no gating and no voting.

- Blend A's weights came from greedy forward selection on an out-of-distribution proxy set: text from 7 local
  open-weight generators plus Project Gutenberg books. That set was used only to evaluate models and choose the
  blend, never for training, and the report discloses this.
- The uploaded runs used seed 42 and 5-fold CV. The weights live in the research repo's `outputs/runs/`, not
  in git.
- `make_submission.py` re-creates all three uploaded CSVs exactly from those runs (`results/reproduction.csv`).
  The full retrain takes about 4 h on two 12 GB GPUs. Don't retrain to "check" anything.

### classical/ — plug-in models

- One model = `models/<name>.py` (subclass of `BaseModel`: `fit`, `predict_proba`) + `configs/<name>.yaml`
  (params and an optional `search:` grid) + one line in `models/__init__.py` `REGISTRY`.
- `python train.py --model <name>` runs the search, fits on the 8k rows and scores the fixed 2k. It writes
  `results/<name>/` and updates `submission_log.csv`.
- `--final` refits on 10k rows and writes `kaggle_submissions/<date>_<name>.csv`.
- Full guide: `classical/CONTRIBUTING.md`.

## Rules for any change

1. Only add or modify files inside `submissions/todo-team/`. The single exception is the pointer
   `AGENTS.md`/`CLAUDE.md` at the repo root on `main`, which must not go into the course PR (see Branches).
2. Never commit competition CSVs (`data/`), Kaggle credentials, `.env`, model weights, checkpoints, `outputs/`,
   `artifacts/` or caches. Run `git status` and check the staged list before every commit. The course root
   `.gitignore` ignores every `*.csv`; our `.gitignore` re-includes only results, logs, the val split and
   the uploaded submissions.
3. Supervised labels come only from `train.csv`. Don't use external LLM APIs to label or score test passages,
   and don't look for hidden labels (RAID or anywhere else).
4. Never type a number into a report or README by hand. Every number comes from a results file written by code:
   - `classical/report_tables.py`, `pretrained/scripts/report_tables.py`, `common/compare_tracks.py` and
     `reports/build_tables.py` generate the LaTeX tables;
   - unknown values stay as visible `[TODO: ...]`.
5. Never tune anything on the 2,000 validation rows or on the Kaggle leaderboard.
6. Small commits, one idea each. Never squash, rebase or force-push pushed history: graders read it. The
   classical track uses the prefixes `feat(<model>)`, `exp(<model>)`, `sub(<model>)` and `log(<model>)`.
7. Ask a human before: any Kaggle upload, opening or updating the course PR, force-pushing, deleting branches,
   or a run longer than about 2 hours.
8. After a Kaggle upload, log the score with `python common/update_submission_log.py ... --public <score>`.
   It rewrites `SUBMISSIONS.md`. Then regenerate the tables.

## Branches

- `main` holds everything; work here or on a short-lived branch merged into `main`.
- `submission/todo-team` is the branch the course PR will be opened from. It equals `main` without the root
  pointer files. To update it:

  ```bash
  # from the repo root
  git checkout submission/todo-team
  git merge main
  git rm -q --ignore-unmatch AGENTS.md CLAUDE.md   # root pointers only; the team folder's copies stay
  git commit -m "Keep root pointer files out of the course PR"
  ```

  After the first removal, later merges keep them out.
- `pretrained`, `classical` and `model/*` are old branches, already merged, and kept for history.

## Regenerating everything (after new results)

```bash
cd submissions/todo-team
python classical/report_tables.py
python pretrained/scripts/report_tables.py
python common/compare_tracks.py
python reports/build_tables.py
# compile reports/kaggle_report.tex (tectonic or Overleaf with the whole team folder uploaded)
```

## Open TODOs (search for `[TODO`)

- Kaggle team name. Rename the folder with `git mv submissions/todo-team submissions/<name>` and rename the
  branch to `submission/<name>`.
- Member names and NetIDs.
- Kaggle scores for C1–C3 (files ready, not uploaded yet).
- Private scores (after the deadline, Nov 11, 2026).
- Which file is the final Kaggle selection.
- The PR URL.
- The AI-use disclosure in `README.md` §6 and report Q6.3: the team must say what they verified.
