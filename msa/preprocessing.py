"""Text cleaning.

Improvements over the paper's pipeline (HTML strip -> special chars -> stem -> stop words):
  * negations are KEPT (the paper's generic stop-word list would delete "not", "no", "never"),
  * negation scope marking ("not good" -> NOT_good) so the model can tell "good" from "not good",
  * contraction expansion (don't -> do not), emoticon tokens, optional Porter stemming.
"""
from __future__ import annotations

import html
import re

import pandas as pd

try:  # optional dependency
    from nltk.stem import PorterStemmer

    _PORTER = PorterStemmer()
except Exception:  # pragma: no cover
    _PORTER = None

NEGATIONS = {"not", "no", "never", "nor", "neither", "none", "nothing", "nowhere",
             "hardly", "barely", "scarcely", "without", "cannot"}
CLAUSE_BREAK = {".", ",", ";", ":", "!", "?"}

_STOP = """a about above after again against all am an and any are as at be because been before being below
between both but by can did do does doing down during each few for from further had has have having he her here
hers herself him himself his how i if in into is it its itself just me more most my myself of off on once only or
other our ours ourselves out over own same she should so some such than that the their theirs them themselves then
there these they this those through to too under until up very was we were what when where which while who whom why
will with would you your yours yourself yourselves br film movie""".split()
STOPWORDS = set(_STOP) - NEGATIONS  # negations must survive

_HTML = re.compile(r"<[^>]+>")
_URL = re.compile(r"https?://\S+|www\.\S+")
_TOKEN = re.compile(r"[a-z]+|[.,;:!?]")
_CONTRACTIONS = [
    (re.compile(r"\bwon't\b"), "will not"), (re.compile(r"\bcan't\b"), "can not"),
    (re.compile(r"n't\b"), " not"), (re.compile(r"'re\b"), " are"), (re.compile(r"'ve\b"), " have"),
    (re.compile(r"'ll\b"), " will"), (re.compile(r"'m\b"), " am"), (re.compile(r"'d\b"), " would"),
    (re.compile(r"'s\b"), ""),
]
_EMO_POS = re.compile(r"(:-?\)|:-?d\b|;-?\)|<3)", re.I)
_EMO_NEG = re.compile(r"(:-?\(|:'\()")


def _light_stem(w: str) -> str:
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    for suf in ("ing", "ed", "ly", "es", "s"):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            return w[: -len(suf)]
    return w


def stem(word: str) -> str:
    return _PORTER.stem(word) if _PORTER is not None else _light_stem(word)


def normalize_tokens(text: str) -> list[str]:
    """Lower-case, strip HTML/URLs, expand contractions, tokenise (keeps . , ; : ! ? as tokens)."""
    text = html.unescape(str(text)).replace("\u2019", "'")
    text = _HTML.sub(" ", text)
    text = _URL.sub(" ", text)
    text = _EMO_POS.sub(" emopos ", text)  # before lower(): ":D" etc.
    text = _EMO_NEG.sub(" emoneg ", text)
    text = text.lower()
    for pat, rep in _CONTRACTIONS:
        text = pat.sub(rep, text)
    text = text.replace("'", "")
    return _TOKEN.findall(text)


def mark_negation(tokens: list[str], window: int = 3) -> list[str]:
    """Prefix the `window` tokens after a negation (until punctuation) with NOT_."""
    out, remaining = [], 0
    for t in tokens:
        if t in CLAUSE_BREAK:
            remaining = 0
            out.append(t)
        elif t in NEGATIONS:
            remaining = window
            out.append(t)
        elif remaining > 0:
            out.append("NOT_" + t)
            remaining -= 1
        else:
            out.append(t)
    return out


def preprocess(text: str, negation: bool = True, do_stem: bool = True) -> str:
    toks = normalize_tokens(text)
    if negation:
        toks = mark_negation(toks)
    result = []
    for t in toks:
        if t in CLAUSE_BREAK:
            continue
        prefix = "NOT_" if t.startswith("NOT_") else ""
        base = t[4:] if prefix else t
        if base in STOPWORDS or len(base) < 2:
            continue
        result.append(prefix + (stem(base) if do_stem else base))
    return " ".join(result)


def preprocess_paper(text: str) -> str:
    """Approximation of the paper: HTML/special-char removal, stemming, generic stop-word removal.
    No negation handling; negation words are dropped like any other stop word."""
    toks = [t for t in normalize_tokens(text) if t not in CLAUSE_BREAK]
    generic_stop = STOPWORDS | NEGATIONS
    return " ".join(stem(t) for t in toks if t not in generic_stop and len(t) > 1)


def prepare_frame(reviews) -> pd.DataFrame:
    """Build the model input frame: raw text + both cleaned variants."""
    s = pd.Series(list(reviews), dtype="object").astype(str)
    return pd.DataFrame({
        "review": s,
        "clean": s.map(preprocess),
        "clean_paper": s.map(preprocess_paper),
    })
