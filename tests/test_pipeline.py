import numpy as np
from msa.data import make_demo_data, split_data
from msa.lexicon import LexiconFeatures
from msa.models import NBSVM, model_zoo
from msa.preprocessing import mark_negation, normalize_tokens, prepare_frame, preprocess, preprocess_paper
from msa.evaluate import mcnemar_exact, bootstrap_ci


def test_negation_survives_and_is_marked():
    out = preprocess("This movie is NOT good. I don't like it!")
    assert "NOT_good" in out.split()
    assert "not" in out.split()


def test_paper_pipeline_drops_negation():
    assert "not" not in preprocess_paper("This is not good").split()


def test_html_and_contractions():
    toks = normalize_tokens("<br />I can't believe it&amp;")
    assert "br" not in toks and "can" in toks and "not" in toks


def test_mark_negation_stops_at_punctuation():
    assert mark_negation(["not", "good", ".", "fine"]) == ["not", "NOT_good", ".", "fine"]


def test_lexicon_flips_negated_words():
    f = LexiconFeatures().transform(["not bad", "bad", "good"])
    assert f[0, 0] > f[0, 1]          # "not bad" -> effective positive
    assert f[1, 1] > f[1, 0]
    assert (f >= 0).all()


def test_nbsvm_fits():
    from scipy import sparse
    rng = np.random.default_rng(0)
    X = sparse.csr_matrix(rng.random((60, 20)))
    y = np.array(["negative", "positive"] * 30)
    assert NBSVM().fit(X, y).predict(X).shape == (60,)


def test_stats_helpers():
    a = np.array([1, 1, 1, 0, 0], bool); b = np.array([1, 0, 0, 0, 0], bool)
    assert 0 <= mcnemar_exact(a, b) <= 1
    lo, hi = bootstrap_ci(a.astype(float)); assert lo <= hi


def test_every_model_trains_and_predicts():
    df = make_demo_data(400)
    tr, te = split_data(df)
    Xtr, Xte = prepare_frame(tr.review), prepare_frame(te.review)
    for name, (pipe, _) in model_zoo(len(Xtr)).items():
        pipe.fit(Xtr, tr.label)
        acc = (pipe.predict(Xte) == te.label.to_numpy()).mean()
        assert acc > 0.6, name
