"""Load a saved run and predict any CSV with a `text` column (default: test.csv).

  python predict.py --run outputs/runs/<run_name>
  python predict.py --run outputs/runs/<run_name> --input data/unlabeled.csv --output preds.csv
  python predict.py --run outputs/runs/<run_name> --device mps   # e.g. model trained on CUDA, used on a Mac

Runs that used `final.strategy: fold_average` are predicted with the mean of
their fold models, which exist only if they were trained with
`output.save_fold_models=true`. Their test predictions are already in
<run>/test_pred.csv either way.
"""
from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
if sys.platform == "darwin":  # LightGBM + PyTorch each ship libomp on macOS; allow both in one process
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import read_yaml, resolve  # noqa: E402
from src.data import load_csv  # noqa: E402
from src.models import get_model_class  # noqa: E402
from src.preprocessing import preprocess  # noqa: E402


def model_dirs(run: Path) -> list[Path]:
    if (run / "model.joblib").exists() or (run / "transformer_hparams.json").exists():
        return [run]
    folds = sorted(p for p in run.glob("fold_*") if p.is_dir())
    if folds:
        return folds
    raise FileNotFoundError(
        f"No saved model in {run}. Fold-averaged runs only keep fold models when trained with "
        f"--set output.save_fold_models=true; their test predictions are in {run / 'test_pred.csv'}.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="outputs/runs/<run_name>")
    ap.add_argument("--input", default=None, help="CSV with id,text (default: config test_file)")
    ap.add_argument("--output", default=None, help="where to write id,label (default: <run>/predictions.csv)")
    ap.add_argument("--device", default=None, help="auto | cuda | cuda:<i> | mps | cpu (torch models only)")
    args = ap.parse_args()

    run = resolve(args.run)
    cfg = read_yaml(run / "config.yaml")
    cls = get_model_class(cfg["model"]["name"])
    df = load_csv(args.input or cfg["data"]["test_file"])
    texts = preprocess(df["text"], cfg["preprocessing"], "preprocess")
    dirs = model_dirs(run)
    preds = []
    for d in dirs:
        model = cls.load(d, device=args.device)
        preds.append(model.predict_proba(texts))
        if hasattr(model, "release"):
            model.release()
    out = pd.DataFrame({"id": df["id"], "label": np.mean(preds, 0)})
    out_path = Path(args.output) if args.output else run / "predictions.csv"
    out.to_csv(out_path, index=False)
    print(f"Wrote {len(out)} predictions ({len(dirs)} model(s)) to {out_path}")


if __name__ == "__main__":
    main()
