"""Model zoo: faithful replicas of the paper's 3 classifiers + improved models."""
from __future__ import annotations

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import VotingClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler
from sklearn.svm import LinearSVC

from .lexicon import LexiconFeatures


class NBSVM(ClassifierMixin, BaseEstimator):
    """NB-weighted logistic regression (Wang & Manning 2012 - reference [9] of the paper).
    Features are scaled by the Naive-Bayes log-count ratio, then a linear model is fitted."""

    def __init__(self, C: float = 4.0, alpha: float = 1.0, max_iter: int = 1000):
        self.C, self.alpha, self.max_iter = C, alpha, max_iter

    def _scale(self, X):
        return sparse.csr_matrix(sparse.csr_matrix(X).multiply(self.r_))

    def fit(self, X, y):
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError("NBSVM supports binary problems only")
        X = sparse.csr_matrix(X)
        p = self.alpha + np.asarray(X[y == self.classes_[1]].sum(axis=0)).ravel()
        q = self.alpha + np.asarray(X[y == self.classes_[0]].sum(axis=0)).ravel()
        self.r_ = np.log((p / p.sum()) / (q / q.sum()))
        self.lr_ = LogisticRegression(C=self.C, max_iter=self.max_iter, solver="liblinear")
        self.lr_.fit(self._scale(X), y)
        return self

    def predict_proba(self, X):
        return self.lr_.predict_proba(self._scale(X))

    def predict(self, X):
        return self.lr_.predict(self._scale(X))

    def decision_function(self, X):
        return self.lr_.decision_function(self._scale(X))


def _features(kind: str, n_train: int, use_lexicon: bool):
    if kind == "paper":
        # paper: CountVectorizer(min_df=0, max_df=1, binary=False, ngram_range=(1,3)) on 7.5k reviews
        min_df = 1 if n_train <= 15000 else 2  # keep memory sane on the full 40k set
        vec, col = CountVectorizer(min_df=min_df, ngram_range=(1, 3)), "clean_paper"
    else:
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.95, sublinear_tf=True, dtype=np.float32)
        col = "clean"
    parts = [("text", vec, col)]
    if use_lexicon and kind != "paper":
        parts.append(("lex", Pipeline([("f", LexiconFeatures()), ("s", MaxAbsScaler())]), "review"))
    return ColumnTransformer(parts, sparse_threshold=1.0)


def _pipe(kind, n_train, lex, clf):
    return Pipeline([("features", _features(kind, n_train, lex)), ("clf", clf)])


def _svm_cal():
    return CalibratedClassifierCV(LinearSVC(C=0.5), cv=3)


def model_zoo(n_train: int, seed: int = 42, use_lexicon: bool = True, binary: bool = True) -> dict:
    """name -> (pipeline, tuning_grid)."""
    zoo = {
        "Paper: Naive Bayes (BoW)": (_pipe("paper", n_train, False, MultinomialNB()), {"clf__alpha": [0.1, 0.5, 1.0]}),
        "Paper: Logistic Reg. (BoW)": (_pipe("paper", n_train, False, LogisticRegression(max_iter=1000, random_state=seed)),
                                       {"clf__C": [0.1, 1, 10]}),
        "Paper: Linear SVM (BoW)": (_pipe("paper", n_train, False, LinearSVC(random_state=seed)), {"clf__C": [0.1, 1, 10]}),
        "Improved: Logistic Reg.": (_pipe("improved", n_train, use_lexicon,
                                          LogisticRegression(C=10, max_iter=2000, random_state=seed)),
                                    {"clf__C": [1, 4, 16, 64]}),
        "Improved: Linear SVM": (_pipe("improved", n_train, use_lexicon, _svm_cal()),
                                 {"clf__estimator__C": [0.1, 0.3, 1.0]}),
        "Improved: Naive Bayes": (_pipe("improved", n_train, use_lexicon, MultinomialNB(alpha=0.3)),
                                  {"clf__alpha": [0.03, 0.1, 0.3, 1.0]}),
    }
    if binary:
        zoo["Improved: NB-SVM"] = (_pipe("improved", n_train, use_lexicon, NBSVM()), {"clf__C": [1, 4, 16]})
        ens = VotingClassifier(
            [("lr", LogisticRegression(C=10, max_iter=2000, random_state=seed)), ("svm", _svm_cal()),
             ("nb", MultinomialNB(alpha=0.3)), ("nbsvm", NBSVM())], voting="soft")
        zoo["Improved: Soft-voting Ensemble"] = (_pipe("improved", n_train, use_lexicon, ens), {})
    return zoo
