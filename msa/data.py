"""Data loading (Kaggle CSV, aclImdb folder, generic CSV, synthetic demo) and splitting."""
from __future__ import annotations

import random
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

TEXT_COLS = ("review", "text", "sentence", "content", "comment")
_POS = {"positive", "pos", "1", "true"}
_NEG = {"negative", "neg", "0", "false"}
_NEU = {"neutral", "neu", "2"}  # only if a 3-class file uses this convention


def _map_label(v) -> str:
    s = str(v).strip().lower()
    if s in _POS:
        return "positive"
    if s in _NEG:
        return "negative"
    if s in _NEU:
        return "neutral"
    raise ValueError(f"Unrecognised label value: {v!r}")


def _rating_to_label(r: pd.Series) -> pd.Series:
    r = pd.to_numeric(r)
    if r.max() > 5:  # 1-10 scale (IMDb)
        return r.map(lambda x: "negative" if x <= 4 else ("neutral" if x <= 6 else "positive"))
    return r.map(lambda x: "negative" if x <= 2 else ("neutral" if x == 3 else "positive"))  # 1-5 scale


def load_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols = {c.lower().strip(): c for c in df.columns}
    tcol = next((cols[c] for c in TEXT_COLS if c in cols), None)
    if tcol is None:
        raise ValueError(f"No text column found in {list(df.columns)}; expected one of {TEXT_COLS}")
    if "sentiment" in cols:
        y = df[cols["sentiment"]].map(_map_label)
    elif "label" in cols:
        y = df[cols["label"]].map(_map_label)
    elif "rating" in cols:
        y = _rating_to_label(df[cols["rating"]])
    else:
        raise ValueError("Need a 'sentiment', 'label' or 'rating' column.")
    out = pd.DataFrame({"review": df[tcol].astype(str), "label": y})
    return out.dropna().drop_duplicates(subset="review").reset_index(drop=True)


def load_aclimdb(root: str | Path) -> pd.DataFrame:
    """Official Stanford split. File names look like '123_7.txt' (id_rating)."""
    root = Path(root)
    rows = []
    for split in ("train", "test"):
        for lab in ("pos", "neg"):
            for f in (root / split / lab).glob("*.txt"):
                m = re.match(r"\d+_(\d+)\.txt", f.name)
                rows.append({"review": f.read_text(encoding="utf8", errors="ignore"),
                             "label": "positive" if lab == "pos" else "negative",
                             "rating": int(m.group(1)) if m else None, "split": split})
    if not rows:
        raise FileNotFoundError(f"No aclImdb files under {root}")
    return pd.DataFrame(rows)


def load_any(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    return load_aclimdb(p) if p.is_dir() else load_csv(p)


def make_demo_data(n: int = 3000, seed: int = 42) -> pd.DataFrame:
    """Synthetic reviews incl. negation, mixed sentiment and label noise. Smoke-test only!"""
    rng = random.Random(seed)
    pos = ["a wonderful story", "brilliant acting", "a beautiful and moving script", "great pacing and a superb cast",
           "truly entertaining from start to end", "an absolute masterpiece", "well written and funny", "I loved it"]
    neg = ["a terrible story", "awful acting", "a boring and predictable script", "poor pacing and a weak cast",
           "painfully dull from start to end", "a complete waste of time", "badly written and lazy", "I hated it"]
    not_pos = ["not good", "not funny", "never entertaining", "not worth watching", "not great"]   # negative
    not_neg = ["not bad", "not boring", "not terrible", "never dull", "not a waste of time"]     # positive
    filler = ["The movie runs about two hours.", "It was shot on location.", "The director has made other films.",
              "The soundtrack is by a well known composer.", "I watched this with friends on a weekend."]
    rows = []
    for _ in range(n):
        label = rng.choice(["positive", "negative"])
        main, opp = (pos, neg) if label == "positive" else (neg, pos)
        neg_phr = not_neg if label == "positive" else not_pos
        parts = [rng.choice(main) for _ in range(rng.randint(1, 2))]
        if rng.random() < 0.3:
            parts.append(rng.choice(neg_phr))
        if rng.random() < 0.3:
            parts.append("although " + rng.choice(opp))
        parts += rng.sample(filler, rng.randint(1, 3))
        rng.shuffle(parts)
        text = ". ".join(p[0].upper() + p[1:] for p in parts) + rng.choice([".", "!", "..."])
        if rng.random() < 0.06:  # label noise
            label = "negative" if label == "positive" else "positive"
        rows.append({"review": "<br />" + text, "label": label})
    return pd.DataFrame(rows)


def split_data(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42, sample: int | None = None):
    """Official split if a 'split' column exists, else stratified. Optional stratified sub-sampling
    (e.g. sample=10000 reproduces the paper's 7500/2500 regime)."""
    if sample and sample < len(df):
        df, _ = train_test_split(df, train_size=sample, stratify=df["label"], random_state=seed)
        df = df.drop(columns=["split"], errors="ignore")
    if "split" in df.columns:
        return (df[df.split == "train"].reset_index(drop=True), df[df.split == "test"].reset_index(drop=True))
    tr, te = train_test_split(df, test_size=test_size, stratify=df["label"], random_state=seed)
    return tr.reset_index(drop=True), te.reset_index(drop=True)
