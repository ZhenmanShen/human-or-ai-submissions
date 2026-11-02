# data/

Competition CSVs go here (download from the Kaggle "Data" tab). They are git-ignored.

```
train.csv  test.csv  unlabeled.csv  train_starter_2000.csv  sample_submission.csv
```

`external/` holds extra labeled data (`text,label,source`; label 1 = AI, 0 = human) created with
`scripts/add_external_data.py` / `scripts/generate_ai_text.py`. Everything in `data/` except this README is
git-ignored (see `.gitignore` to change that for `external/`).
