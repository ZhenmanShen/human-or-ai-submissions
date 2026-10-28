"""TF-IDF -> linear SVM (hinge loss). The SVM margin is squashed with a
sigmoid to [0, 1]; AUROC only needs the ranking so this is sufficient.
Set calibrate: true for Platt-scaled probabilities (slower, 3x fits)."""
from __future__ import annotations

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC

from ..features import TfidfFeaturizer
from .base import BaseModel


class TfidfSVM(BaseModel):
    name = "tfidf_svm"

    def __init__(self, tfidf: dict, C: float = 0.5, calibrate: bool = False,
                 fit_vocab_on_unlabeled: bool = False):
        super().__init__(tfidf=tfidf, C=C, calibrate=calibrate)
        self.featurizer = TfidfFeaturizer(**tfidf)
        self.fit_vocab_on_unlabeled = fit_vocab_on_unlabeled
        svm = LinearSVC(C=C, max_iter=5000)
        self.clf = CalibratedClassifierCV(svm, cv=3) if calibrate else svm
        self.calibrate = calibrate

    def fit(self, texts, y, val_texts=None, val_y=None, unlabeled_texts=None):
        vocab_texts = list(texts) + (list(unlabeled_texts) if self.fit_vocab_on_unlabeled and unlabeled_texts is not None else [])
        self.featurizer.fit(vocab_texts)
        X = self.featurizer.transform(texts)
        print(f"  features: {X.shape[1]:,}  |  fitting linear SVM ...")
        self.clf.fit(X, y)
        return self

    def predict_proba(self, texts):
        X = self.featurizer.transform(texts)
        if self.calibrate:
            return self.clf.predict_proba(X)[:, 1]
        return 1.0 / (1.0 + np.exp(-self.clf.decision_function(X)))
