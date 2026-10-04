# Movie Sentiment Plus

Improved, reproducible re-implementation of *"Sentimental Analysis of Movie Reviews Using Machine Learning"*
(ITM Web Conf. 53, 02006, 2023) with negation-aware preprocessing, TF-IDF, a lexicon hybrid, NB-SVM,
an ensemble, statistical testing and a prediction CLI. See `docs/IMPROVEMENTS.md` for the paper audit and full change list.

## Setup
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run
```bash
# 1. smoke test, no data needed (synthetic; numbers are NOT meaningful)
python -m msa.train --demo

# 2. real data (see data/README.md), full dataset
python -m msa.train --data "data/IMDB Dataset.csv" --out outputs/full

# 3. paper's regime (7.5k train / 2.5k test) for an apples-to-apples comparison
python -m msa.train --data "data/IMDB Dataset.csv" --sample 10000 --out outputs/paper_regime

# options: --tune (grid search)  --cv 5  --no-lexicon (ablation)  --only "NB-SVM" "Ensemble"  --seed 7
```
Outputs in `--out`: `results.csv`, `report.md`, `mcnemar.csv`, `confusion_matrices.png`, `roc_curves.png`,
`model_comparison.png`, `top_features.csv`, `error_analysis.csv`, `best_model.joblib`.

## Predict
```bash
python -m msa.predict "Not bad at all, I loved it!" "A boring waste of time." --model outputs/full/best_model.joblib
streamlit run app.py        # optional UI
```

## Test
```bash
pip install pytest && pytest -q
```
Optional deep-learning baseline: `python -m msa.transformer_baseline --data "data/IMDB Dataset.csv"`.

## Layout
```
msa/preprocessing.py  negation-aware cleaning (+ paper-style cleaning for the replica)
msa/lexicon.py        lexicon / style features (the paper's missing hybrid idea)
msa/models.py         paper replica models + improved models + NB-SVM + ensemble
msa/evaluate.py       metrics, bootstrap CI, McNemar, plots, top features, error analysis
msa/train.py          full experiment CLI        msa/predict.py  inference CLI
msa/data.py           CSV / aclImdb loaders, splits, synthetic demo data
tests/                unit + end-to-end tests    docs/IMPROVEMENTS.md  audit and change log
```
