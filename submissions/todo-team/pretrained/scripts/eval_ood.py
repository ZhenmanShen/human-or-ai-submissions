"""Score runs (and blends) on the held-out evaluation sets built by scripts/build_eval_set.py.

  python scripts/eval_ood.py --runs <runA> <runB> ...           # runs trained with data.eval_files set
  python scripts/eval_ood.py --runs <A> <B> --blend best=<A>:2,<B>:1
  python scripts/eval_ood.py --all                              # every run in outputs/runs with eval files

Metrics (one row per run/blend, appended/updated in experiments/ood_eval.csv):
  cv_auc           stratified OOF AUROC on train (in-distribution reference)
  newgen_auc       train HUMAN passages (out-of-fold scores) vs AI text from unseen generators on train's
                   domains (eval_newgen). Each AI row is scored by the CV fold model whose validation fold
                   holds the human passage it was generated from, so no model scores text derived from
                   something it trained on. The fairest "unseen generator" number we have.
  newgen_tpr1      share of those AI rows caught at the threshold that flags 1% of train humans (OOF)
  newgen_min_gen   AUROC on the hardest single generator (newgen_<generator> columns have each one)
  creative_auc     held-out Gutenberg human poems/stories vs AI poems/stories (eval_creative, fold-mean
                   scores); archaic human vs modern AI, so it overstates separability -- use it to rank
                   models, not as a test-score estimate. poem/story columns split it by form.
  creative_fpr5    share of Gutenberg human passages flagged as AI at the threshold that flags 5% of train
                   humans -> do we mistake (old) poetry/fiction for AI? Lower is better.
  proxy_auc        ONE pooled, weighted AUROC meant to track the Kaggle score: Kaggle ranks all 3,000 test
                   passages together, so a model must also put different domains/generators on one scale.
                   Humans: train OOF (1 - creative_share) + Gutenberg poems/stories (creative_share).
                   AI: train OOF (seen generators) and eval_newgen (unseen) split by --unseen-share over the
                   non-creative mass, + AI poems/stories (creative_share). creative_share = 0.41 is the
                   creative-writing cluster's share of test (AGENTS.md, topic shift); the generator mix of
                   test is unknown, so proxy_auc_u25 / _u75 repeat it with 25% / 75% unseen.
Blends average each model's scores after mapping them through that model's OOF-score CDF on train
(monotone, so single-model AUROCs are unchanged; puts all models on one scale).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import resolve  # noqa: E402
from src.inference import find_run  # noqa: E402
from src.utils import auroc  # noqa: E402

OUT = "experiments/ood_eval.csv"
NEWGEN, CREATIVE = "data/eval/eval_newgen.csv", "data/eval/eval_creative.csv"
CREATIVE_SHARE, UNSEEN_SHARE = 0.41, 0.5  # proxy_auc mix, see the docstring


def thr_at_fpr(human_scores, fpr):
    return float(np.quantile(human_scores, 1 - fpr))


class RunScores:
    """A run's OOF scores on train and its scores on the eval sets (seed-matched for eval_newgen)."""

    def __init__(self, run: Path, meta_newgen: pd.DataFrame, meta_creative: pd.DataFrame):
        self.name = run.name
        oof = pd.read_csv(run / "oof.csv")
        oof = oof[oof["fold"] >= 0]
        self.oof, self.y, self.ref = oof["pred"].to_numpy(), oof["label"].to_numpy(), np.sort(oof["pred"].to_numpy())
        fold_of = dict(zip(oof["id"], oof["fold"]))
        ng = pd.read_csv(run / "eval_eval_newgen.csv")
        assert (ng["id"].to_numpy() == meta_newgen["id"].to_numpy()).all(), f"{run.name}: eval_newgen rows differ"
        folds = meta_newgen["seed_id"].map(fold_of)
        if folds.isna().any():
            raise ValueError(f"{run.name}: {int(folds.isna().sum())} eval_newgen seeds have no OOF fold")
        self.newgen = np.array([ng[f"pred_f{int(f)}"].iat[i] for i, f in enumerate(folds)])
        cr = pd.read_csv(run / "eval_eval_creative.csv")
        assert (cr["id"].to_numpy() == meta_creative["id"].to_numpy()).all(), f"{run.name}: eval_creative rows differ"
        self.creative = cr["pred"].to_numpy()

    def cdf(self, x):
        """Map scores to their quantile among this run's OOF scores (for blending)."""
        return np.searchsorted(self.ref, x, side="right") / len(self.ref)


def blend(name, members: list[tuple[RunScores, float]]):
    b = RunScores.__new__(RunScores)
    b.name = name
    w = np.array([wi for _, wi in members], float)
    w /= w.sum()
    b.y = members[0][0].y
    for attr in ("oof", "newgen", "creative"):
        setattr(b, attr, sum(wi * m.cdf(getattr(m, attr)) for (m, _), wi in zip(members, w)))
    b.ref = np.sort(b.oof)
    return b


def metrics(r: RunScores, meta_ng: pd.DataFrame, meta_cr: pd.DataFrame) -> dict:
    human = r.oof[r.y == 0]
    row = {"run": r.name, "cv_auc": auroc(r.y, r.oof)}
    yy = np.r_[np.zeros(len(human)), np.ones(len(r.newgen))]
    row["newgen_auc"] = auroc(yy, np.r_[human, r.newgen])
    row["newgen_tpr1"] = float((r.newgen > thr_at_fpr(human, 0.01)).mean())
    for g in sorted(meta_ng["generator"].unique()):
        m = (meta_ng["generator"] == g).to_numpy()
        row[f"newgen_{g}"] = auroc(np.r_[np.zeros(len(human)), np.ones(m.sum())], np.r_[human, r.newgen[m]])
    row["newgen_min_gen"] = min(v for k, v in row.items() if k.startswith("newgen_") and k[7:] in set(meta_ng["generator"]))
    yc = meta_cr["label"].to_numpy()
    row["creative_auc"] = auroc(yc, r.creative)
    for form, hmode, amodes in [("poem", "poetry", ("poem", "poem_rw")), ("story", "stories", ("story",))]:
        m = ((meta_cr["mode"] == hmode) | meta_cr["mode"].isin(amodes)).to_numpy()
        row[f"creative_{form}_auc"] = auroc(yc[m], r.creative[m])
    row["creative_fpr5"] = float((r.creative[yc == 0] > thr_at_fpr(human, 0.05)).mean())
    for tag, unseen in [("", UNSEEN_SHARE), ("_u25", 0.25), ("_u75", 0.75)]:
        row[f"proxy_auc{tag}"] = proxy_auc(r, yc, CREATIVE_SHARE, unseen)
    return row


def proxy_auc(r: RunScores, yc: np.ndarray, creative: float, unseen: float) -> float:
    """Weighted AUROC over all pools at once (see the module docstring); every pool's weights sum to
    its share of its class, so pool sizes don't matter."""
    pools = [  # (scores, label, share of that label's mass)
        (r.oof[r.y == 0], 0, 1 - creative),
        (r.creative[yc == 0], 0, creative),
        (r.oof[r.y == 1], 1, (1 - creative) * (1 - unseen)),
        (r.newgen, 1, (1 - creative) * unseen),
        (r.creative[yc == 1], 1, creative),
    ]
    s = np.concatenate([p for p, _, _ in pools])
    y = np.concatenate([np.full(len(p), lab) for p, lab, _ in pools])
    w = np.concatenate([np.full(len(p), share / len(p)) for p, _, share in pools])
    return float(roc_auc_score(y, s, sample_weight=w))


def subset(r: RunScores, ng_mask: np.ndarray, cr_mask: np.ndarray) -> RunScores:
    s = RunScores.__new__(RunScores)
    s.name, s.y, s.oof, s.ref = r.name, r.y, r.oof, r.ref
    s.newgen, s.creative = r.newgen[ng_mask], r.creative[cr_mask]
    return s


def greedy(cands: dict[str, RunScores], yc: np.ndarray, steps: int = 12) -> dict[str, int]:
    """Caruana selection with replacement, maximising proxy_auc; returns {run: times picked}."""
    picked: list[str] = []
    best = -1.0
    for _ in range(steps):
        scored = []
        for k in cands:
            trial = picked + [k]
            b = blend("t", [(cands[m], trial.count(m)) for m in dict.fromkeys(trial)])
            scored.append((proxy_auc(b, yc, CREATIVE_SHARE, UNSEEN_SHARE), k))
        auc, k = max(scored)
        if auc <= best + 1e-5:
            break
        best, picked = auc, picked + [k]
    return {m: picked.count(m) for m in dict.fromkeys(picked)}


def greedy_check(cands: dict[str, RunScores], meta_ng, meta_cr) -> list[dict]:
    """Pick a greedy blend on half of the generators (and half of the Gutenberg humans), score it on the
    other half, then swap: an honest estimate of what greedy selection on the proxy is worth."""
    gens = sorted(meta_ng["generator"].unique())
    yc = meta_cr["label"].to_numpy()
    rng = np.random.default_rng(0)
    human_half = rng.random(len(meta_cr)) < 0.5
    rows = []
    for half in (0, 1):
        g_sel = set(gens[half::2])
        masks = {}
        for side, g in (("sel", g_sel), ("held", set(gens) - g_sel)):
            hum = human_half if (side == "sel") == (half == 0) else ~human_half
            masks[side] = (meta_ng["generator"].isin(g).to_numpy(),
                           np.where(yc == 0, hum, meta_cr["generator"].isin(g).to_numpy()))
        sel = {k: subset(r, *masks["sel"]) for k, r in cands.items()}
        held = {k: subset(r, *masks["held"]) for k, r in cands.items()}
        w = greedy(sel, yc[masks["sel"][1]])
        b = blend("g", [(held[m], c) for m, c in w.items()])
        best_single = max(cands, key=lambda k: proxy_auc(sel[k], yc[masks["sel"][1]], CREATIVE_SHARE, UNSEEN_SHARE))
        rows.append({"selected_on": ",".join(sorted(g_sel)), "weights": w,
                     "held_blend": proxy_auc(b, yc[masks["held"][1]], CREATIVE_SHARE, UNSEEN_SHARE),
                     "held_best_single": proxy_auc(held[best_single], yc[masks["held"][1]], CREATIVE_SHARE,
                                                   UNSEEN_SHARE), "best_single": best_single})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="*", default=[])
    ap.add_argument("--all", action="store_true", help="every run in outputs/runs that has the eval files")
    ap.add_argument("--blend", nargs="*", default=[], help="name=runA:w,runB:w (runs must be in --runs/--all)")
    ap.add_argument("--greedy", action="store_true",
                    help="greedy blend on proxy_auc (+ a select-on-half / score-on-other-half check)")
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args()

    meta_ng, meta_cr = pd.read_csv(resolve(NEWGEN)), pd.read_csv(resolve(CREATIVE))
    runs = [find_run(r) for r in args.runs]
    if args.all:
        runs += [p for p in sorted(resolve("outputs/runs").iterdir())
                 if (p / "eval_eval_newgen.csv").exists() and (p / "eval_eval_creative.csv").exists() and p not in runs]
    scores = {}
    for p in runs:
        try:
            s = RunScores(p, meta_ng, meta_cr)
            scores[s.name] = scores[p.name.split("__")[-1]] = s  # blends may use the short job name
        except (AssertionError, ValueError, FileNotFoundError, KeyError) as e:
            print(f"skip {p.name}: {e}")
    rows = [metrics(s, meta_ng, meta_cr) for k, s in scores.items() if k == s.name]
    for spec in args.blend:
        name, members = spec.split("=", 1)
        parts = [(scores[m.rsplit(":", 1)[0]], float(m.rsplit(":", 1)[1]) if ":" in m else 1.0)
                 for m in members.split(",")]
        rows.append(metrics(blend(f"blend:{name}", parts), meta_ng, meta_cr))
    if args.greedy:
        cands = {k: s for k, s in scores.items() if k == s.name}
        for c in greedy_check(cands, meta_ng, meta_cr):
            print(f"greedy check: select on [{c['selected_on']}] -> held-out proxy {c['held_blend']:.4f} "
                  f"(best single {c['best_single'].split('__')[-1]}: {c['held_best_single']:.4f}) {c['weights']}")
        w = greedy(cands, meta_cr["label"].to_numpy())
        spec = ",".join(f"{m.split('__')[-1]}:{c}" for m, c in w.items())
        print(f"greedy blend on all eval data: {spec}")
        rows.append(metrics(blend("blend:greedy_proxy", [(cands[m], c) for m, c in w.items()]), meta_ng, meta_cr))
    res = pd.DataFrame(rows)
    lead = ["run", "proxy_auc", "cv_auc", "newgen_auc", "newgen_tpr1", "newgen_min_gen", "creative_auc",
            "creative_poem_auc", "creative_story_auc", "creative_fpr5"]
    res = res[lead + [c for c in res.columns if c not in lead]]
    print(res[lead].sort_values("proxy_auc", ascending=False).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    if not args.no_save:
        out = resolve(OUT)
        if out.exists():  # replace rows of the same run/blend, keep the others
            old = pd.read_csv(out)
            res = pd.concat([old[~old["run"].isin(res["run"])], res], ignore_index=True)
        res.round(5).to_csv(out, index=False)
        print(f"Saved {out}")


if __name__ == "__main__":
    main()
