# Movie Sentiment Analyzer — VS Code to Render

## 1. Open in VS Code
Open this folder:
`movie_sentiment_deploy_ready`

The real dataset is already in:
`data/IMDB Dataset.csv`

## 2. Create and activate virtual environment (Windows PowerShell)
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If you only have Python 3.13:
```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 3. Verify the project
```powershell
pytest -q
```
Expected: 8 passed.

## 4. First smoke test (no real dataset)
```powershell
python -m msa.train --demo
```

## 5. Train on your real IMDb dataset
### Recommended first run: paper-style 10,000 reviews
This is much faster and gives a reproducible experiment:
```powershell
python -m msa.train --data "data/IMDB Dataset.csv" --sample 10000 --cv 3 --out outputs/paper_regime
```

For a faster first run:
```powershell
python -m msa.train --data "data/IMDB Dataset.csv" --sample 10000 --cv 0 --out outputs/paper_regime
```

### Full 50,000-review experiment
```powershell
python -m msa.train --data "data/IMDB Dataset.csv" --cv 3 --out outputs/full
```

The full run can take considerably longer because multiple models and preprocessing are evaluated.

## 6. Check generated files
Look inside:
`outputs/paper_regime/`

Important:
- `best_model.joblib`
- `results.csv`
- `report.md`
- `confusion_matrices.png`
- `roc_curves.png`
- `model_comparison.png`
- `top_features.csv`
- `error_analysis.csv`

## 7. Run the Streamlit UI locally
```powershell
streamlit run app.py
```

The browser will open a local Streamlit address. If it does not, copy the URL printed in the terminal.

The app automatically searches:
1. `outputs/full/best_model.joblib`
2. `outputs/paper_regime/best_model.joblib`
3. `outputs/best_model.joblib`

## 8. Test the CLI
```powershell
python -m msa.predict "Not bad at all, I loved it!" "A boring waste of time." --model outputs/paper_regime/best_model.joblib
```

## 9. GitHub
Create a GitHub repository and push the project.

Do NOT commit `.venv`, `__pycache__`, or temporary files. Add this to `.gitignore`:
```gitignore
.venv/
__pycache__/
*.pyc
.pytest_cache/
.streamlit/secrets.toml
```

The dataset is ~27 MB, so it is below GitHub's normal 100 MB single-file limit.

The trained `best_model.joblib` may be much larger. If it becomes too large for normal GitHub storage, use Git LFS or a model-storage service instead of forcing it into Git.

## 10. Render deployment
The included `render.yaml` uses:
```text
pip install -r requirements-render.txt
streamlit run app.py --server.port $PORT --server.address 0.0.0.0
```

Important: Render should NOT train the model on every deployment.

Before deployment, commit your trained model at:
`outputs/paper_regime/best_model.joblib`

Then either:
- deploy from the GitHub repository using the included `render.yaml`, or
- create a Render Web Service manually with the same build/start commands.

The Render environment variable:
`MODEL_PATH=outputs/paper_regime/best_model.joblib`

## 11. If GitHub rejects the model file
Use Git LFS:
```powershell
git lfs install
git lfs track "outputs/*/best_model.joblib"
git add .gitattributes
git add outputs/paper_regime/best_model.joblib
git commit -m "Add trained sentiment model"
git push
```

## 12. Project explanation for viva
Pipeline:
Dataset → HTML/URL cleaning → contraction expansion → negation handling →
TF-IDF + lexicon/style features → ML models → evaluation → saved Joblib model →
Streamlit UI → Render deployment.

The project compares paper-style Naive Bayes, Logistic Regression and Linear SVM
with improved Logistic Regression, Linear SVM, Naive Bayes, NB-SVM and a soft-voting ensemble.
