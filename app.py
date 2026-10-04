"""Streamlit UI for the trained Movie Sentiment Analyzer."""
from pathlib import Path
import os
import joblib
import streamlit as st
from msa.predict import predict_texts

st.set_page_config(page_title="Movie Sentiment Analyzer", page_icon="🎬", layout="centered")

st.markdown("""
<style>
.main {max-width: 900px; margin: auto;}
.hero {padding: 1.5rem; border-radius: 18px; background: linear-gradient(135deg,#111827,#374151);
       color: white; margin-bottom: 1.2rem;}
.hero h1 {margin-bottom: .3rem;}
.card {padding: 1rem 1.2rem; border-radius: 14px; border: 1px solid #ddd; margin-top: 1rem;}
.small {color:#6b7280; font-size:.9rem;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
<h1>🎬 Movie Review Sentiment Analyzer</h1>
<p>Negation-aware NLP + TF-IDF + lexicon features + machine learning</p>
</div>
""", unsafe_allow_html=True)

def find_model():
    env = os.getenv("MODEL_PATH")
    candidates = [env] if env else []
    candidates += [
        "outputs/full/best_model.joblib",
        "outputs/paper_regime/best_model.joblib",
        "outputs/best_model.joblib",
    ]
    for p in candidates:
        if p and Path(p).exists():
            return p
    return None

model_path = find_model()
if not model_path:
    st.error("No trained model found. Train the model first, then restart the app.")
    st.code('python -m msa.train --data "data/IMDB Dataset.csv" --sample 10000 --cv 0 --out outputs/paper_regime')
    st.stop()

@st.cache_resource
def load_model(path):
    return joblib.load(path)

model = load_model(model_path)

st.caption(f"Loaded model: `{model_path}`")

band = st.sidebar.slider(
    "Uncertain/neutral band",
    min_value=0.0, max_value=0.4, value=0.15, step=0.01,
    help="A review close to 50% positive probability is marked uncertain. IMDb itself has no neutral class."
)

examples = {
    "Positive": "The acting was brilliant and the story was beautiful. I loved every minute of it.",
    "Negative": "The movie was boring, predictable and a complete waste of time.",
    "Negation": "It was not bad at all. The performances were surprisingly good."
}

choice = st.selectbox("Try an example", ["Custom"] + list(examples))
default_text = "" if choice == "Custom" else examples[choice]

text = st.text_area(
    "Enter a movie review",
    value=default_text,
    height=180,
    placeholder="Example: The story was excellent and the acting was wonderful..."
)

if st.button("🔎 Analyze Sentiment", type="primary", use_container_width=True):
    if not text.strip():
        st.warning("Please enter a review.")
    else:
        label, p = predict_texts(model, [text], band)[0]
        st.markdown('<div class="card">', unsafe_allow_html=True)
        if label == "positive":
            st.success("### 😊 Positive")
        elif label == "negative":
            st.error("### 😞 Negative")
        else:
            st.warning("### 😐 Uncertain / Neutral-like")
        st.metric("Positive probability", f"{p:.1%}")
        st.progress(float(p))
        st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<p class="small">Note: “uncertain/neutral” is a confidence heuristic, not a trained neutral class.</p>',
            unsafe_allow_html=True)
