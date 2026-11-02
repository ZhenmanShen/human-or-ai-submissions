"""Shared evaluation helpers, so both tracks compute every reported number the same way."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

# Word-count bin edges used in both reports (Q4.2): <150, 150-250, >250 words.
LENGTH_BINS = (0, 150, 250, np.inf)
LENGTH_LABELS = ("<150 words", "150-250 words", ">250 words")


def auroc(y_true, y_score) -> float:
    return float(roc_auc_score(y_true, y_score))


def tpr_at_fpr(y_true, y_score, max_fpr: float = 0.01) -> float:
    """Highest true-positive rate reachable with false-positive rate <= max_fpr."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(tpr[fpr <= max_fpr].max())


def summary(y_true, y_score) -> dict:
    return {"auroc": auroc(y_true, y_score), "tpr_at_1fpr": tpr_at_fpr(y_true, y_score, 0.01)}


def word_count(texts) -> np.ndarray:
    return np.array([len(str(t).split()) for t in texts])


def is_verse(text: str, max_words_per_line: float = 12.0, min_lines: int = 4) -> bool:
    """Line-structure heuristic for poetry: at least `min_lines` non-empty lines and a
    median of at most `max_words_per_line` words per line."""
    lines = [ln for ln in str(text).splitlines() if ln.strip()]
    if len(lines) < min_lines:
        return False
    return float(np.median([len(ln.split()) for ln in lines])) <= max_words_per_line


def subgroup_auroc(df: pd.DataFrame, group_col: str, y_col: str = "label", score_col: str = "pred") -> pd.DataFrame:
    """Count, class balance and AUROC per subgroup (AUROC is NaN when a group has one class)."""
    rows = []
    for g, d in df.groupby(group_col, observed=True, sort=True):
        both = d[y_col].nunique() == 2
        rows.append({"group": g, "count": len(d), "n_ai": int(d[y_col].sum()),
                     "auroc": auroc(d[y_col], d[score_col]) if both else np.nan})
    return pd.DataFrame(rows)


def length_bin_auroc(texts, y_true, y_score) -> pd.DataFrame:
    df = pd.DataFrame({"label": np.asarray(y_true), "pred": np.asarray(y_score)})
    df["bin"] = pd.cut(word_count(texts), LENGTH_BINS, labels=LENGTH_LABELS, right=False)
    return subgroup_auroc(df, "bin")


def verse_prose_auroc(texts, y_true, y_score) -> pd.DataFrame:
    df = pd.DataFrame({"label": np.asarray(y_true), "pred": np.asarray(y_score)})
    df["form"] = ["verse" if is_verse(t) else "prose" for t in texts]
    return subgroup_auroc(df, "form")


def errors(ids, y_true, y_score, threshold: float = 0.5, k: int = 5) -> pd.DataFrame:
    """The k most confident false positives (human, score >= threshold) and false negatives
    (AI, score < threshold), with their predicted P(AI)."""
    df = pd.DataFrame({"id": np.asarray(ids), "label": np.asarray(y_true), "p_ai": np.asarray(y_score)})
    fp = df[(df.label == 0) & (df.p_ai >= threshold)].sort_values("p_ai", ascending=False).head(k)
    fn = df[(df.label == 1) & (df.p_ai < threshold)].sort_values("p_ai").head(k)
    return pd.concat([fp.assign(error="FP"), fn.assign(error="FN")], ignore_index=True)
