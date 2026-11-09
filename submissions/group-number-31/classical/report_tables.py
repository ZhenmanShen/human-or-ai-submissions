"""Turn results/ into the LaTeX table rows and figures that reports/kaggle_classical_report.tex inputs.

  python report_tables.py

Writes reports/tables/*.tex (table bodies) and reports/figures/*.pdf from results/*/metrics.json,
results/*/search.csv, results/experiments/*.csv, results/ood_eval.csv and ../submission_log.csv. The FP/FN hypotheses are the
one hand-written input (reports/fp_fn_notes_stylo_gbm.yaml). Nothing is typed by hand.
"""
from __future__ import annotations

import json
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


OOD_ROWS = [  # (ood_eval.csv row, label, Kaggle upload in submission_log.csv)
    ("tfidf_lr@v1", "C1 v1: word + char\\_wb TF-IDF", "final_E_tfidf_lr@v1"),
    ("tfidf_lr", "C1 v2: + raw char, distorted word n-grams", "final_E_tfidf_lr"),
    ("transformer_scratch@v1", "C3 v1: raw text", "final_F_transformer_scratch@v1"),
    ("transformer_scratch", "C3 v2: lines joined, text distorted", "final_F_transformer_scratch"),
    ("stylo_gbm", "C2 (unchanged, reference)", "final_D_stylo_gbm"),
]


def q5_ood():
    """C1/C3 before and after the shift-robust changes: fixed split, OOD proxy (ood_eval.py), Kaggle public."""
    o = pd.read_csv(RES / "ood_eval.csv").set_index("model")
    log = pd.read_csv(TEAM / "submission_log.csv", dtype=str, keep_default_na=False).set_index("model")
    rows = []
    for name, label, sub in OOD_ROWS:
        r = o.loc[name]
        pub = log.loc[sub, "kaggle_public"] if sub in log.index else "--"
        pub = f"{float(pub):.5f}" if pub.replace(".", "", 1).isdigit() else tex(pub)
        rows.append(f"{label} & {r.val_auc:.4f} & {r.newgen_auc:.4f} & {r.creative_auc:.4f} & {r.creative_fpr5:.2f} & "
                    f"{r.proxy_auc:.4f} & {pub} \\\\")
    write("q53_ood.tex", rows)


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
    for step in ("q12", "q13", "q2", "q3", "q4", "q5", "q5_ood", "appendix"):
        globals()[step]()
