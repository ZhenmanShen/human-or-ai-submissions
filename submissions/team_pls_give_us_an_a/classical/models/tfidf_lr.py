"""C1: word (1-2)-gram + character (2-5)-gram TF-IDF -> L2 logistic regression.

The course baseline family (a word TF-IDF + LR reference gets ~0.868 private AUROC). Character n-grams
within word boundaries add sub-word style signal (suffixes, punctuation habits, homoglyph-broken words).
Vocabulary and IDF are fit on the training rows only (the sklearn regex tokenizer, no pretrained tokenizer).
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from .base import BaseModel


class TfidfLR(BaseModel):
    name = "tfidf_lr"

    def __init__(self, C: float = 4.0, word_ngrams=(1, 2), char_ngrams=(2, 5), word_min_df: int = 2,
                 char_min_df: int = 3, max_features: int = 300000, lowercase: bool = True, seed: int = 0):
        self.C, self.seed = C, seed
        self.word = TfidfVectorizer(analyzer="word", ngram_range=tuple(word_ngrams), min_df=word_min_df,
                                    max_features=max_features, sublinear_tf=True, lowercase=lowercase)
        self.char = TfidfVectorizer(analyzer="char_wb", ngram_range=tuple(char_ngrams), min_df=char_min_df,
                                    max_features=max_features, sublinear_tf=True, lowercase=lowercase)

    def _X(self, texts, fit=False):
        f = (lambda v: v.fit_transform(texts)) if fit else (lambda v: v.transform(texts))
        return sp.hstack([f(self.word), f(self.char)]).tocsr()

    def fit(self, texts, y, unlabeled_texts=None):
        X = self._X(texts, fit=True)
        self.clf = LogisticRegression(C=self.C, solver="liblinear", random_state=self.seed).fit(X, y)
        return self

    def predict_proba(self, texts):
        return self.clf.predict_proba(self._X(texts))[:, 1].astype(np.float64)
