"""Metrics, significance tests, plots and error analysis."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score, roc_curve)


def compute_metrics(y_true, y_pred, proba=None, classes=None) -> dict:
    m = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }
    if proba is not None and classes is not None:
        try:
            m["roc_auc"] = (roc_auc_score(y_true, proba[:, list(classes).index("positive")])
                            if len(classes) == 2 else roc_auc_score(y_true, proba, multi_class="ovr", labels=classes))
        except Exception:
            m["roc_auc"] = np.nan
    return m


def bootstrap_ci(correct: np.ndarray, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05):
    """95 % bootstrap CI of accuracy from a 0/1 'correct' vector."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(correct), size=(n_boot, len(correct)))
    accs = correct[idx].mean(axis=1)
    return float(np.quantile(accs, alpha / 2)), float(np.quantile(accs, 1 - alpha / 2))


def mcnemar_exact(correct_a: np.ndarray, correct_b: np.ndarray) -> float:
    """Exact two-sided McNemar p-value: do models A and B differ significantly on the same test set?"""
    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))
    return 1.0 if b + c == 0 else float(binomtest(b, b + c, 0.5).pvalue)


def plot_confusions(results: dict, classes, path):
    n = len(results)
    cols = min(4, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.6 * cols, 3.4 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (name, r) in zip(axes.ravel(), results.items()):
        ax.axis("on")
        cm = confusion_matrix(r["y_true"], r["y_pred"], labels=classes)
        ax.imshow(cm, cmap="Blues")
        for i in range(len(classes)):
            for j in range(len(classes)):
                ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set_xticks(range(len(classes)), classes, rotation=30, fontsize=7)
        ax.set_yticks(range(len(classes)), classes, fontsize=7)
        ax.set_title(name, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_roc(results: dict, path):
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for name, r in results.items():
        if r.get("score") is None:
            continue
        fpr, tpr, _ = roc_curve(r["y_true"] == "positive", r["score"])
        ax.plot(fpr, tpr, label=name)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curves")
    ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_comparison(df: pd.DataFrame, path):
    d = df.sort_values("accuracy")
    fig, ax = plt.subplots(figsize=(7, 0.45 * len(d) + 1.5))
    colors = ["#999999" if n.startswith("Paper") else "#2a7ab9" for n in d.index]
    err = [d["accuracy"] - d["acc_ci_low"], d["acc_ci_high"] - d["accuracy"]]
    ax.barh(d.index, d["accuracy"], xerr=err, color=colors, capsize=3)
    ax.set_xlim(max(0, d["acc_ci_low"].min() - 0.05), min(1, d["acc_ci_high"].max() + 0.03))
    ax.set(xlabel="Test accuracy (95% bootstrap CI)", title="Grey = paper replica, blue = improved")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def top_features(pipe, n: int = 25) -> pd.DataFrame | None:
    """Most positive / negative features of a linear model (interpretability)."""
    clf = pipe.named_steps["clf"]
    if hasattr(clf, "lr_"):
        coef = clf.lr_.coef_.ravel() * clf.r_
    elif hasattr(clf, "coef_"):
        coef = clf.coef_.ravel()
    else:
        return None
    names = pipe.named_steps["features"].get_feature_names_out()
    order = np.argsort(coef)
    neg = pd.DataFrame({"feature": names[order[:n]], "weight": coef[order[:n]], "pushes_towards": "negative"})
    pos = pd.DataFrame({"feature": names[order[::-1][:n]], "weight": coef[order[::-1][:n]], "pushes_towards": "positive"})
    return pd.concat([pos, neg], ignore_index=True)


def error_analysis(test_df: pd.DataFrame, y_pred, proba, classes, n: int = 50) -> pd.DataFrame:
    wrong = np.asarray(y_pred) != test_df["label"].to_numpy()
    out = test_df.loc[wrong, ["review", "label"]].copy()
    out["predicted"] = np.asarray(y_pred)[wrong]
    if proba is not None:
        out["confidence"] = proba[wrong].max(axis=1)
        out = out.sort_values("confidence", ascending=False)
    out["review"] = out["review"].str.slice(0, 400)
    return out.head(n)
