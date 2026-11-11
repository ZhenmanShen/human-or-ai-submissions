"""Turn results/ into the LaTeX table rows and figures that reports/kaggle_classical_report.tex inputs.

  python report_tables.py

Writes reports/tables/*.tex (table bodies) and reports/figures/*.pdf from results/*/metrics.json,
results/*/search.csv, results/experiments/*.csv, results/selection/ (topic-held-out CV), results/topic_cv/ and
../submission_log.csv. The FP/FN hypotheses are the
one hand-written input (reports/fp_fn_notes_stylo_gbm.yaml). Nothing is typed by hand.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402

HERE = Path(__file__).resolve().parent
TEAM = HERE.parent
RES, EXP = HERE / "results", HERE / "results" / "experiments"
TAB, FIG = HERE / "reports" / "tables", HERE / "reports" / "figures"

ROWS = [  # (results dir, label, representation)
    ("tfidf_lr", "C1 \\texttt{tfidf\\_lr}",
     "word 1--2 + raw char 1--5 + distorted-text word 1--3 TF-IDF $\\to$ logistic regression"),
    ("stylo_gbm", "C2 \\texttt{stylo\\_gbm}", "101 hand-crafted stylometric features $\\to$ LightGBM"),
    ("transformer_scratch", "C3 \\texttt{transformer\\_scratch}",
     "lines joined, text distorted, 12k byte-level BPE $\\to$ 4-layer Transformer encoder, random init"),
    ("linear_svm", "\\texttt{linear\\_svm}", "C1 v1 features (word + char\\_wb) $\\to$ linear SVM"),
    ("nbsvm", "\\texttt{nbsvm}", "binary n-grams scaled by NB log-count ratio $\\to$ LR"),
    ("rf_stylo", "\\texttt{rf\\_stylo}", "C2 features $\\to$ random forest"),
    ("ensemble_c123", "\\texttt{ensemble\\_c123}", "C1+C2+C3 CDF blend, weights from inner-CV OOF"),
]


def tex(s) -> str:
    return str(s).replace("_", "\\_").replace("%", "\\%").replace("&", "\\&").replace("#", "\\#")


def f4(x) -> str:
    return "--" if x is None or x == "" or (isinstance(x, float) and np.isnan(x)) else f"{float(x):.4f}"


def secs(x) -> str:
    x = float(x)
    return f"{x:.0f} s" if x < 120 else f"{x / 60:.1f} min"


def metrics(name: str) -> dict:
    return json.loads((RES / name / "metrics.json").read_text(encoding="utf-8"))


def write(name: str, rows: list[str]) -> None:
    TAB.mkdir(parents=True, exist_ok=True)
    (TAB / name).write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"wrote reports/tables/{name}")


def short_hw(h: str) -> str:
    return "CPU" if h.startswith("CPU") else h.replace("NVIDIA GeForce ", "")


def q12():
    rows = []
    for name, label, rep in ROWS:
        m = metrics(name)
        rows.append(f"{label} & {rep} & {f4(m['train_auroc'])} & {f4(m['val_auroc'])} & "
                    f"{secs(m['fit_time_s'])} ({short_hw(m['hardware'])}) \\\\")
        if name == "transformer_scratch":
            rows.append("\\midrule")
    write("q12_models.tex", rows)
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    d = sorted(((metrics(n)["val_auroc"], n) for n, _, _ in ROWS))
    ax.barh([n for _, n in d], [v for v, _ in d], color="#236B82")
    ax.set_xlim(0.95, 1.0)
    ax.set_xlabel("validation AUROC (fixed 2,000 rows)")
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "val_auroc_bar.pdf")
    plt.close(fig)


def q13():
    rows = []
    for name in ("tfidf_lr", "stylo_gbm", "linear_svm", "nbsvm", "rf_stylo"):
        s = pd.read_csv(RES / name / "search.csv")
        keys = [c for c in s.columns if c not in ("cv_auroc_mean", "cv_auroc_std", "cv_auroc_pooled", "time_s", "chosen")]
        grid = " $\\times$ ".join(f"{tex(k)} \\{{{', '.join(sorted(map(str, s[k].unique()), key=lambda v: (len(v), v)))}\\}}"
                                  for k in keys)
        best = s[s.chosen].iloc[0]
        chosen = ", ".join(f"{tex(k)}={best[k]}" for k in keys)
        rows.append(f"\\texttt{{{tex(name)}}} & {grid} ({len(s)} settings) & {chosen}; "
                    f"inner CV {best.cv_auroc_mean:.4f} $\\pm$ {best.cv_auroc_std:.4f} \\\\")
    write("q13_hpsearch.tex", rows)


def q2():
    d = pd.read_csv(EXP / "learning_curve.csv")
    main, other = d[d.model == "stylo_gbm"], d[d.model == "tfidf_lr"].set_index("train_rows")
    write("q21_curve.tex", [f"{int(r.train_rows):,} & {f4(r.train_auroc)} & {f4(r.val_auroc)} & "
                            f"TF-IDF LR val {f4(other.loc[r.train_rows, 'val_auroc'])} \\\\" for _, r in main.iterrows()])
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    for model, c in [("stylo_gbm", "#236B82"), ("tfidf_lr", "#C0763A")]:
        s = d[d.model == model]
        ax.plot(s.train_rows, s.train_auroc, "--o", color=c, ms=4, label=f"{model} train")
        ax.plot(s.train_rows, s.val_auroc, "-o", color=c, ms=4, label=f"{model} validation")
    ax.set_xscale("log")
    ax.set_xticks([500, 1000, 2000, 5000, 8000], ["500", "1k", "2k", "5k", "8k"])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("training rows (nested subsets of the fixed 8,000)")
    ax.set_ylabel("AUROC")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "learning_curve.pdf")
    plt.close(fig)


def q3():
    d = pd.read_csv(EXP / "semi_supervised.csv")
    rows = []
    for _, r in d.iterrows():
        what = "Supervised only" if r.rounds == 0 else f"Self-training, {int(r.rounds)} round{'s' if r.rounds > 1 else ''}"
        rows.append(f"{what} (\\texttt{{{tex(r.model)}}}) & {r.val_auroc:.5f} \\\\")
    write("q32_semisup.tex", rows)


def q4():
    s = pd.read_csv(EXP / "subgroups_stylo_gbm.csv")
    write("q42_subgroups.tex", [f"{tex(r.group).replace('<', '$<$').replace('>', '$>$')} ({r.grouping}; {int(r.n_ai)} AI) & "
                                f"{int(r['count'])} & {r.auroc:.5f} \\\\" for _, r in s.iterrows()])
    e = pd.read_csv(EXP / "fp_fn_stylo_gbm.csv")
    notes = yaml.safe_load((HERE / "reports" / "fp_fn_notes_stylo_gbm.yaml").read_text(encoding="utf-8"))
    write("q43_errors.tex", [f"\\mbox{{{int(r.id)}}} & {r.error} & {r.p_ai:.3f} & {tex(notes.get(int(r.id), '[TODO]'))} \\\\"
                             for _, r in e.iterrows()])


def q5():
    rows = []
    for name, label in [("stylo_gbm", "Baseline: C2 \\texttt{stylo\\_gbm} alone"),
                        ("ensemble_c123", "Improvement: C1+C2+C3 blend, inner-CV OOF weights"),
                        ("stylo_gbm@no_added", "Ablation: C2 without the 9 added features"),
                        ("stylo_gbm@no_layout", "Ablation: C2 without the 8 layout features")]:
        m = metrics(name)
        rows.append(f"{label} & {m['val_auroc']:.5f} & {secs(m['fit_time_s'])} & TPR@1\\%FPR {m['val_tpr_at_1fpr']:.3f} \\\\")
    write("q53_improvement.tex", rows)


VERSIONS = [  # (label, Kaggle upload row in submission_log.csv, selection run, its candidate, results dir)
    ("Baseline: C1 v1 (word + in-word char TF-IDF)", "final_E_tfidf_lr@v1", "tfidf_lr", "v1+C64+case", "tfidf_lr"),
    ("Improvement: C1 v2 (+ raw char, distorted n-grams)", "final_E_tfidf_lr", "tfidf_lr", "raw_d300+C64+lc",
     "tfidf_lr"),
    ("Baseline: C3 v1 (raw text)", "final_F_transformer_scratch@v1", "transformer_scratch", "v1",
     "transformer_scratch"),
    ("Improvement: C3 v2 (lines joined, distorted)", "final_F_transformer_scratch", "transformer_scratch", "d300_join",
     "transformer_scratch"),
]


def candidates(model: str) -> pd.DataFrame:
    """results/selection/<model>/candidates.csv (select_design.py): topic / stratified inner CV, fixed val."""
    return pd.read_csv(RES / "selection" / model / "candidates.csv").set_index("candidate")


def old_metrics(commit: str, model: str) -> dict:
    """results/<model>/metrics.json as of `commit` (the team folder was renamed since, so find its path)."""
    files = subprocess.run(["git", "ls-tree", "-r", "--full-tree", "--name-only", commit], cwd=HERE, capture_output=True,
                           text=True).stdout.split()
    path = next(f for f in files if f.endswith(f"classical/results/{model}/metrics.json"))
    return json.loads(subprocess.run(["git", "show", f"{commit}:{path}"], cwd=HERE, capture_output=True,
                                     text=True).stdout)


def q5_versions():
    """Q5.3 rows for C1/C3 v1 vs v2: val and fit time from the version's own validation run (v1: from the commit of
    the v1 upload), topic-held-out inner CV from results/selection/, Kaggle public from the log."""
    log = pd.read_csv(TEAM / "submission_log.csv", dtype=str, keep_default_na=False).set_index("model")
    rows = []
    for label, sub, sel, cand, model in VERSIONS:
        m = old_metrics(log.loc[sub, "commit"], model) if sub.endswith("@v1") else metrics(model)
        c = candidates(sel).loc[cand]
        strat = f"; stratified CV {c.stratified_mean:.4f}" if "stratified_mean" in c else ""
        rows.append(f"{label} & {m['val_auroc']:.5f} & {secs(m['fit_time_s'])} & topic-held-out CV {c.topic_mean:.4f} "
                    f"$\\pm$ {c.topic_std:.4f}{strat}; Kaggle public {float(log.loc[sub, 'kaggle_public']):.5f} \\\\")
    write("q53_versions.tex", rows)


def q5_topics():
    """Per held-out topic fold: AUROC of C1/C3 v1 and v2 (topic CV folds; topics from results/topic_cv/clusters.csv)."""
    cl = pd.read_csv(RES / "topic_cv" / "clusters.csv")
    cols = [(sel, cand) for _, _, sel, cand, _ in VERSIONS]
    folds = {k: [float(x) for x in candidates(k[0]).loc[k[1], "topic_folds"].split()] for k in cols}
    rows = []
    for f, g in cl.groupby("fold", sort=True):
        topics = "; ".join(f"{' '.join(t.split()[:3])} ({int(n):,})" for t, n in zip(g.top_terms, g.n))
        ai = (g.ai_share * g.n).sum() / g.n.sum()
        rows.append(f"{f + 1}: {tex(topics)} & {ai:.2f} & " + " & ".join(f"{folds[k][f]:.4f}" for k in cols) + " \\\\")
    rows.append("\\midrule")
    rows.append("Mean $\\pm$ s.d.\\ over the 5 folds & & " + " & ".join(
        f"{np.mean(folds[k]):.4f} $\\pm$ {np.std(folds[k]):.4f}" for k in cols) + " \\\\")
    write("q53_topics.tex", rows)


def q13_selection():
    """C1 / C3 design selection (select_design.py): per view, the best setting by topic CV, with the reported-only
    stratified CV and fixed-split AUROC, and the winner under the pre-registered rule."""
    for model in ("tfidf_lr", "transformer_scratch"):
        p = RES / "selection" / model / "selection.json"
        if not p.exists():
            print(f"skip q13_selection {model}: no {p.relative_to(HERE)}")
            continue
        sel = json.loads(p.read_text(encoding="utf-8"))
        c = candidates(model)
        c["group"] = [n.split("+")[0] for n in c.index]
        rows = []
        for g, d in c.groupby("group", sort=False):
            r = d.loc[d.topic_mean.idxmax()]
            mark = (" (winner)" if r.name == sel["winner"] else "") + (" (best)" if r.name == sel["best"] else "")
            strat = f"{r.stratified_mean:.4f}" if "stratified_mean" in r else "--"
            rows.append(f"\\texttt{{{tex(r.name)}}}{mark} & {r.topic_mean:.4f} $\\pm$ {r.topic_std:.4f} & {strat} & "
                        f"{r.val_auroc:.4f} & {int(d.tied.sum())}/{len(d)} \\\\")
        if sel["winner"] not in [x for x in c.loc[c.groupby("group", sort=False).topic_mean.idxmax()].index]:
            r = c.loc[sel["winner"]]
            strat = f"{r.stratified_mean:.4f}" if "stratified_mean" in r else "--"
            rows.append(f"\\texttt{{{tex(r.name)}}} (winner) & {r.topic_mean:.4f} $\\pm$ {r.topic_std:.4f} & {strat} & "
                        f"{r.val_auroc:.4f} & tied \\\\")
        write(f"q13_selection_{model}.tex", rows)


BREAK = chr(92) + 'allowbreak{}'  # long model names may wrap at underscores


def appendix():
    log = pd.read_csv(TEAM / "submission_log.csv", dtype=str, keep_default_na=False).sort_values(["date", "track", "model"])
    url = "https://github.com/ZhenmanShen/human-or-ai-submissions/commit/"
    rows = []
    for _, r in log.iterrows():
        commit = f"\\href{{{url}{r.commit}}}{{\\texttt{{{r.commit[:7]}}}}}" if r.commit else ""
        rows.append(f"{r.date} & {r.track} & \\texttt{{{tex(r.model).replace('_', '_' + BREAK)}}} & {commit} & {f'{float(r.val_auroc):.5f}' if r.val_auroc else '--'} & "
                    f"{tex(r.kaggle_public)} & {tex(r.kaggle_private) or '--'} \\\\")
    write("appendix_submissions.tex", rows)


if __name__ == "__main__":
    for step in ("q12", "q13", "q13_selection", "q2", "q3", "q4", "q5", "q5_versions", "q5_topics", "appendix"):
        globals()[step]()
