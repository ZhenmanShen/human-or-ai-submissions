"""Score classical models on held-out out-of-distribution text, as a proxy for the Kaggle test set.

Test differs from train: unseen generators, and ~15-17% verse against 1.6% of train. The fixed 2,000-row
validation split cannot show this (every C1-C3 model is >= 0.96 there, but their public scores spread from
0.83 to 0.96). This script re-uses the pretrained track's evaluation sets (pretrained/scripts/build_eval_set.py;
eval only, never trained on):
  eval_newgen.csv    AI rewrites of train human passages by 7 generators that are not in train. Only rows whose
                     source passage (seed_id) is in the validation split are used, so no model has trained on
                     the original.
  eval_creative.csv  human poems/stories from held-out Project Gutenberg books vs AI poems/stories.

Each model is fit on the 8,000 training rows (as train.py does) and scored on the 2,000 validation rows plus
those two sets. proxy_auc is one pooled AUROC with sample weights, mixed like pretrained/scripts/eval_ood.py:
creative 41% of each class, the rest of the AI mass split 50/50 between validation AI (seen generators) and
eval_newgen (unseen). Hyperparameters are still chosen by inner CV on the training rows only; this proxy is
used to compare model designs, never to tune on the validation rows or the leaderboard.

  python ood_eval.py --model tfidf_lr                        # params from results/<model>/metrics.json
  python ood_eval.py --model tfidf_lr --set C=8 --tag c8     # a variant
Writes results/ood_eval.csv (one row per model/tag, updated in place).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "common"))
sys.path.insert(0, str(HERE))
from data import SEED, load_csv, resolve_data_dir, train_val_split  # noqa: E402
from train import build, load_config, parse_sets  # noqa: E402

OUT = HERE / "results" / "ood_eval.csv"
CREATIVE_SHARE, UNSEEN_SHARE = 0.41, 0.5


def eval_sets(data_dir, val_ids) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = resolve_data_dir(data_dir) / "eval"
    if not (d / "eval_newgen.csv").exists():
        raise SystemExit(f"{d} has no eval_newgen.csv; build it with pretrained/scripts/build_eval_set.py")
    newgen = pd.read_csv(d / "eval_newgen.csv")
    return newgen[newgen["seed_id"].isin(set(val_ids))].reset_index(drop=True), pd.read_csv(d / "eval_creative.csv")


def proxy(val: pd.DataFrame, p_val, newgen: pd.DataFrame, p_new, creative: pd.DataFrame, p_cre) -> dict:
    y = np.concatenate([val["label"], np.ones(len(newgen)), creative["label"]])
    s = np.concatenate([p_val, p_new, p_cre])
    w = np.concatenate([
        np.where(val["label"] == 1, (1 - CREATIVE_SHARE) * (1 - UNSEEN_SHARE) / (val["label"] == 1).sum(),
                 (1 - CREATIVE_SHARE) / (val["label"] == 0).sum()),
        np.full(len(newgen), (1 - CREATIVE_SHARE) * UNSEEN_SHARE / max(1, len(newgen))),
        np.where(creative["label"] == 1, CREATIVE_SHARE / (creative["label"] == 1).sum(),
                 CREATIVE_SHARE / (creative["label"] == 0).sum()),
    ])
    hum_val = np.asarray(p_val)[val["label"].to_numpy() == 0]
    thr = np.quantile(hum_val, 0.95)  # flags 5% of validation humans
    cre_h = np.asarray(p_cre)[creative["label"].to_numpy() == 0]
    return {
        "val_auc": roc_auc_score(val["label"], p_val),
        "newgen_auc": roc_auc_score(np.r_[np.zeros(len(hum_val)), np.ones(len(newgen))], np.r_[hum_val, p_new]),
        "creative_auc": roc_auc_score(creative["label"], p_cre),
        "creative_fpr5": float((cre_h > thr).mean()),
        "proxy_auc": roc_auc_score(y, s, sample_weight=w),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--device", default=None)
    ap.add_argument("--data-dir", default=None)
    args = ap.parse_args(argv)

    cfg = load_config(args.model)
    name = args.model + (f"@{args.tag}" if args.tag else "")
    mfile = HERE / "results" / args.model / "metrics.json"
    params = json.loads(mfile.read_text(encoding="utf-8"))["params"] if mfile.exists() else dict(cfg["params"])
    params = {**params, **parse_sets(args.set)}
    train, val = train_val_split(args.data_dir)
    newgen, creative = eval_sets(args.data_dir, val["id"])
    unl = load_csv("unlabeled", args.data_dir)["text"].tolist()
    model = build(cfg, params, args.seed, args.device).fit(train["text"].tolist(), train["label"].to_numpy(),
                                                          unlabeled_texts=unl)
    p = [model.predict_proba(df["text"].tolist()) for df in (val, newgen, creative)]
    row = {"model": name, **proxy(val, p[0], newgen, p[1], creative, p[2]), "n_newgen": len(newgen),
           "params": json.dumps(params, default=str)}
    print("  ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}" for k, v in row.items() if k != "params"))
    df = pd.read_csv(OUT) if OUT.exists() else pd.DataFrame(columns=list(row))
    df = pd.concat([df[df["model"] != name], pd.DataFrame([row])], ignore_index=True)
    df.to_csv(OUT, index=False, lineterminator="\n", float_format="%.5f")


if __name__ == "__main__":
    main()
