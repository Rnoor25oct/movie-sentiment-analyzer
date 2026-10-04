"""Small sentiment lexicon + negation-aware feature extractor.

The paper's introduction promises a hybrid rule-based + ML approach but never implements it.
This module supplies that missing piece: lexicon counts (with negation flipping) are appended
to the TF-IDF features. All features are non-negative so Naive Bayes can use them as well.
"""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from .preprocessing import CLAUSE_BREAK, NEGATIONS, normalize_tokens

POSITIVE = set("""good great excellent amazing wonderful brilliant superb fantastic fabulous outstanding perfect
beautiful love loved loves lovely enjoy enjoyed enjoyable entertaining fun funny hilarious charming delightful
masterpiece gem touching moving powerful gripping engaging compelling thrilling stunning terrific impressive
clever witty heartwarming memorable satisfying solid strong best better favorite recommend recommended worthwhile
captivating riveting flawless genius magnificent refreshing uplifting sweet warm talented gorgeous incredible
remarkable splendid winner entertained pleasure fresh original authentic realistic emotional inspiring
phenomenal spectacular nice pleasant likable""".split())

NEGATIVE = set("""bad worse worst awful terrible horrible dreadful poor boring bored dull tedious slow predictable
disappointing disappointed disappointment waste wasted pointless stupid silly ridiculous lame weak mess messy
annoying irritating painful unwatchable forgettable mediocre bland cheap clich\u00e9 cliche cliched laughable
hate hated hates dislike failure flop fails failed fail garbage trash rubbish junk nonsense incoherent confusing
amateurish wooden flat lifeless unconvincing overrated underwhelming unfunny cringe obnoxious pathetic
tiresome lazy sloppy shallow contrived dreary uninteresting insulting offensive abysmal atrocious disaster
nasty ugly unbearable avoid regret""".split())

FEATURE_NAMES = ["lex_eff_pos", "lex_eff_neg", "lex_pos_ratio", "lex_exclaim", "lex_question",
                 "lex_caps_ratio", "lex_log_len", "lex_negation_count"]


class LexiconFeatures(BaseEstimator, TransformerMixin):
    """Turn raw review text into 8 dense, non-negative lexicon / style features."""

    def __init__(self, window: int = 3):
        self.window = window

    def fit(self, X, y=None):
        return self

    def _one(self, text: str) -> list[float]:
        toks = normalize_tokens(text)
        words = [t for t in toks if t not in CLAUSE_BREAK]
        pos = neg = npos = nneg = nneg_words = 0
        remaining = 0
        for t in toks:
            if t in CLAUSE_BREAK:
                remaining = 0
                continue
            if t in NEGATIONS:
                remaining = self.window
                nneg_words += 1
                continue
            negated = remaining > 0
            remaining = max(0, remaining - 1)
            if t in POSITIVE or t == "emopos":
                pos += 1
                npos += negated
            elif t in NEGATIVE or t == "emoneg":
                neg += 1
                nneg += negated
        eff_pos = (pos - npos) + nneg      # "not bad" counts as positive
        eff_neg = (neg - nneg) + npos      # "not good" counts as negative
        raw = str(text)
        caps = sum(1 for w in raw.split() if len(w) >= 3 and w.isupper() and w.isalpha())
        n = max(len(words), 1)
        return [eff_pos, eff_neg, eff_pos / (eff_pos + eff_neg + 1.0), raw.count("!"), raw.count("?"),
                caps / n, float(np.log1p(n)), nneg_words]

    def transform(self, X):
        return np.asarray([self._one(t) for t in np.asarray(X).ravel()], dtype=float)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(FEATURE_NAMES, dtype=object)
