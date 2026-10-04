"""Predict from a trained model:  python -m msa.predict "Not bad at all!" "Dull and boring."  """
from __future__ import annotations

import argparse

import joblib
import numpy as np

from .preprocessing import prepare_frame


def predict_texts(model, texts, uncertain_band: float = 0.15):
    """Return (label, P(positive)). If |P(pos) - 0.5| < band the label is 'neutral/uncertain'.
    NOTE: this is a confidence heuristic, NOT a trained neutral class (IMDb data has no neutral labels)."""
    X = prepare_frame(texts)
    proba = model.predict_proba(X)
    p_pos = proba[:, list(model.classes_).index("positive")]
    labels = np.where(np.abs(p_pos - 0.5) < uncertain_band, "neutral/uncertain",
                      np.where(p_pos >= 0.5, "positive", "negative"))
    return list(zip(labels.tolist(), p_pos.round(3).tolist()))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("texts", nargs="+")
    ap.add_argument("--model", default="outputs/best_model.joblib")
    ap.add_argument("--band", type=float, default=0.15)
    a = ap.parse_args(argv)
    model = joblib.load(a.model)
    for t, (lab, p) in zip(a.texts, predict_texts(model, a.texts, a.band)):
        print(f"{lab:18s} P(pos)={p:.3f}  | {t[:80]}")


if __name__ == "__main__":
    main()
