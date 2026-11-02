"""Compare our models with GPTZero on the same passages.  DRAFT: the API call is not written yet.

  python scripts/gptzero_compare.py --input data/test.csv --limit 200 --runs <runA> <runB>
  python scripts/gptzero_compare.py --input data/eval/<eval_set>.csv --runs <runA> --submission outputs/submissions/x.csv

Steps:
1. Score each passage of --input with GPTZero (cached in outputs/gptzero/cache.jsonl by text hash, so a
   passage is never paid for twice; --limit takes a fixed random sample to keep API costs down).
2. Score the same passages with each --runs model (saved model, see make_submission.py) and/or take the
   scores from a --submission file (test.csv ids only).
3. Report, per model: Spearman correlation with GPTZero, agreement of the 0.5-thresholded labels, and our
   AUROC when GPTZero's label is used as the reference. If --input has true labels (train / eval sets),
   also each model's and GPTZero's real AUROC — the only number that says which is actually better.
   Use passages the runs never trained on (test, or a data.eval_files set): on train.csv the saved
   models are scored in-sample and look perfect.
   Writes outputs/gptzero/compare_<input>.csv (per passage) and prints a summary table.

Note: this sends the passages to a third-party service. Agreement with GPTZero is a useful sanity check,
not ground truth — GPTZero is itself a classifier with its own blind spots (e.g. older/small LMs).

To finish: implement query_gptzero() below and export GPTZERO_API_KEY=<key>.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import resolve  # noqa: E402
from src.data import load_csv  # noqa: E402
from src.inference import find_run, predict_run  # noqa: E402
from src.utils import auroc, ensure_dir  # noqa: E402

OUT_DIR = "outputs/gptzero"


def query_gptzero(text: str, api_key: str) -> dict:
    """Return GPTZero's verdict for one passage as {"prob_ai": float in [0, 1], "raw": <full response>}.

    TODO (left open on purpose): call the GPTZero API here, e.g. with `requests.post(<endpoint>,
    headers={"x-api-key": api_key, ...}, json={"document": text})`, take the document-level probability
    that the text is AI-generated from the response, and return it with the raw JSON.
    Check the current endpoint / field names in GPTZero's API docs before filling this in.
    """
    raise NotImplementedError("query_gptzero() is not implemented yet - see its docstring")


def text_key(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def load_cache(path: Path) -> dict:
    cache = {}
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    cache[rec["key"]] = rec
    return cache


def gptzero_scores(texts, api_key: str, cache_path: Path, sleep: float) -> np.ndarray:
    cache = load_cache(cache_path)
    todo = [t for t in dict.fromkeys(texts) if text_key(t) not in cache]
    print(f"GPTZero: {len(texts) - len(todo)} cached, {len(todo)} to query")
    with open(cache_path, "a", encoding="utf-8") as f:
        for i, t in enumerate(todo, 1):
            res = query_gptzero(t, api_key)
            rec = {"key": text_key(t), "prob_ai": float(res["prob_ai"]), "raw": res.get("raw")}
            f.write(json.dumps(rec) + "\n")
            f.flush()  # keep what was paid for even if a later call fails
            cache[rec["key"]] = rec
            if i % 25 == 0:
                print(f"  {i}/{len(todo)}")
            time.sleep(sleep)
    return np.array([cache[text_key(t)]["prob_ai"] for t in texts])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default="data/test.csv", help="CSV with id,text[,label]")
    ap.add_argument("--runs", nargs="*", default=[], help="runs (dirs or names) to compare with GPTZero")
    ap.add_argument("--submission", nargs="*", default=[], help="submission CSVs (id,label) to compare")
    ap.add_argument("--limit", type=int, default=None, help="score a random sample of this many passages")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sleep", type=float, default=0.5, help="seconds between API calls")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    api_key = os.environ.get("GPTZERO_API_KEY", "")
    if not api_key:
        sys.exit("Set GPTZERO_API_KEY first (never commit the key).")
    df = load_csv(args.input)
    if args.limit and args.limit < len(df):
        df = df.sample(args.limit, random_state=args.seed).reset_index(drop=True)
    out_dir = ensure_dir(OUT_DIR)
    scores = pd.DataFrame({"id": df["id"]})
    scores["gptzero"] = gptzero_scores(df["text"].tolist(), api_key, out_dir / "cache.jsonl", args.sleep)
    for r in args.runs:
        run = find_run(r)
        scores[run.name] = predict_run(run, df["text"], args.device)
    for s in args.submission:
        sub = pd.read_csv(resolve(s)).set_index("id")["label"]
        scores[Path(s).stem] = sub.reindex(df["id"]).to_numpy()
    if "label" in df:
        scores["label"] = df["label"].to_numpy()

    gz = scores["gptzero"].to_numpy()
    gz_label = (gz >= 0.5).astype(int)
    rows = []
    for c in [c for c in scores.columns if c not in ("id", "label")]:
        p = scores[c].to_numpy(dtype=float)
        ok = ~np.isnan(p)
        row = {"model": c, "n": int(ok.sum())}
        if c != "gptzero":
            row["spearman_vs_gptzero"] = spearmanr(p[ok], gz[ok]).statistic
            row["agreement@0.5"] = float(((p[ok] >= 0.5) == gz_label[ok]).mean())
            if 0 < gz_label[ok].sum() < ok.sum():
                row["auroc_vs_gptzero_labels"] = auroc(gz_label[ok], p[ok])
        if "label" in scores:
            row["auroc_true_labels"] = auroc(scores["label"].to_numpy()[ok], p[ok])
        rows.append(row)
    summary = pd.DataFrame(rows)
    stem = Path(args.input).stem
    scores.to_csv(out_dir / f"compare_{stem}.csv", index=False)
    summary.to_csv(out_dir / f"summary_{stem}.csv", index=False)
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"Per-passage scores: {out_dir / f'compare_{stem}.csv'}")


if __name__ == "__main__":
    main()
