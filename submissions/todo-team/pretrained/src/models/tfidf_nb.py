"""Naive Bayes family on n-gram features.

variant:
  multinomial : MultinomialNB on TF-IDF (lecture-style baseline)
  complement  : ComplementNB (often better on text)
  nbsvm       : NB-weighted logistic regression (Wang & Manning, 2012): scale
                features by the NB log-count ratio, then fit a linear model.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB, MultinomialNB

from ..features import TfidfFeaturizer
from .base import BaseModel


class TfidfNaiveBayes(BaseModel):
    name = "tfidf_nb"

    def __init__(self, tfidf: dict, variant: str = "nbsvm", alpha: float = 1.0, C: float = 4.0):
        super().__init__(tfidf=tfidf, variant=variant, alpha=alpha, C=C)
        self.featurizer = TfidfFeaturizer(**tfidf)
        self.variant, self.alpha, self.C = variant, alpha, C

    def fit(self, texts, y, val_texts=None, val_y=None, unlabeled_texts=None):
        y = np.asarray(y)
        self.featurizer.fit(texts)
        X = self.featurizer.transform(texts)
        print(f"  features: {X.shape[1]:,}  |  fitting {self.variant} ...")
        if self.variant == "multinomial":
            self.clf = MultinomialNB(alpha=self.alpha).fit(X, y)
        elif self.variant == "complement":
            self.clf = ComplementNB(alpha=self.alpha).fit(X, y)
        elif self.variant == "nbsvm":
            p = self.alpha + X[y == 1].sum(0)
            q = self.alpha + X[y == 0].sum(0)
            self.r = np.asarray(np.log((p / p.sum()) / (q / q.sum()))).ravel()
            self.clf = LogisticRegression(C=self.C, max_iter=2000, solver="liblinear")
            self.clf.fit(X @ sp.diags(self.r), y)
        else:
            raise ValueError(f"Unknown variant {self.variant}")
        return self

    def predict_proba(self, texts):
        X = self.featurizer.transform(texts)
        if self.variant == "nbsvm":
            X = X @ sp.diags(self.r)
        return self.clf.predict_proba(X)[:, 1]
