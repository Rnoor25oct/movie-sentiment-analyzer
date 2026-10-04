# Paper review and what this project improves

Paper: *Sentimental Analysis of Movie Reviews Using Machine Learning*, Sharma, Pangaonkar, Gunjan, Rokade,
ITM Web of Conferences 53, 02006 (ICDSIA-2023).

## A. Problems found in the paper

| # | Issue | Evidence in the paper |
|---|-------|-----------------------|
| 1 | **Reported SVM accuracy does not match its own confusion matrix.** | Abstract/Table 1 say SVM = 73 %. Fig. 5 gives (896+882)/2500 = **71.1 %**. Naive Bayes (Fig. 6) is also (916+862)/2500 = **71.1 %**, i.e. SVM and NB *tie*. Logistic Regression (Fig. 4) = (1000+740)/2500 = **69.6 %**. |
| 2 | **Text contradicts the figures on precision.** | Text: "SVM has the highest precision for positive reviews, LR the highest for negative". Figs. 4-6: positive precision LR 0.77 > NB 0.74 > SVM 0.73; negative precision SVM 0.70 > NB 0.69 > LR 0.65. Both claims are reversed. |
| 3 | **Neutral class claimed but never modelled.** | Abstract: "positive, negative or neutral". The IMDb 50K dataset has only positive/negative labels; all reports are 2-class. |
| 4 | **Inconsistent dataset size.** | Abstract: "over 25,000 reviews"; Sect. 3.1: "almost 50,000"; Fig. 2 shows only 7,500 train + 2,500 test = 10,000 used. |
| 5 | **Promised "novel hybrid" method is absent.** | Introduction: rule-based + ML with domain-specific fine-tuning "outperforms both". No such experiment exists. |
| 6 | **70-73 % is low for this dataset.** | TF-IDF + linear models on the full IMDb set typically reach ~88-90 %; the paper uses 20 % of the data and 1.4 M n-gram features for 7.5 k documents (heavy over-parameterisation). |
| 7 | **Pre-processing hurts sentiment.** | Generic stop-word removal deletes "not", "no", "never"; stemming/number removal are fine, but there is no negation handling although the literature review discusses it. |
| 8 | **Weak experimental protocol.** | One random split, no seed, no cross-validation, no hyper-parameter search (despite "parameters were adjusted"), no confidence intervals, no significance test; a 2-point gap on 2,500 samples is within noise. |
| 9 | **Explanations are speculative.** | "SVM might have been able to...", "likely that..." - no error analysis or feature inspection backs the claims. |
| 10 | No ROC/AUC, no MCC, no runtime, no model persistence or deployable code. |

## B. What this project adds (and why)

| Area | Improvement | Where |
|------|-------------|-------|
| Reproducibility | Faithful **paper replica** (CountVectorizer 1-3-grams, same preprocessing style, NB/LR/SVM) so the baseline is measured by the same code as the new models. `--sample 10000` reproduces the 7.5k/2.5k regime. | `models.py`, `train.py` |
| Honest protocol | Fixed seeds, stratified split (or official aclImdb split), **model selection by cross-validation, never on the test set**, optional grid search (`--tune`). | `data.py`, `train.py` |
| Pre-processing | **Negation-aware**: negations kept, scope marked (`not good` -> `NOT_good`), contraction expansion, emoticons, URL/HTML stripping, Porter stemming (if NLTK present) or a light fallback stemmer. | `preprocessing.py` |
| Features | **TF-IDF** (sublinear tf, 1-2-grams, `min_df=2`, `max_df=0.95`) instead of raw counts with 1.4 M features. | `models.py` |
| Hybrid approach | Implements the paper's **missing rule-based + ML idea**: negation-flipped lexicon scores and style features (!, ?, CAPS, length) appended to TF-IDF. Ablation flag `--no-lexicon`. | `lexicon.py` |
| Models | Calibrated linear SVM, tuned LR, NB, **NB-SVM** (Wang & Manning - the paper's ref. [9]), and a **soft-voting ensemble**. | `models.py` |
| Evaluation | Accuracy, macro P/R/F1, **ROC-AUC, MCC**, **95 % bootstrap CI**, **exact McNemar test** vs. the best paper-replica model, confusion matrices, ROC curves, training time. | `evaluate.py` |
| Interpretability | Top positive/negative features of the linear model; **error analysis** CSV of the most confident mistakes. | `evaluate.py` |
| "Neutral" handling | Honest alternative to the unsupported neutral class: a confidence band around P(pos)=0.5 flags *uncertain/neutral* reviews (a heuristic, labelled as such). Generic CSV loader also supports real 3-class data via a `rating`/`label` column. | `predict.py`, `data.py` |
| Deployment | Saved model (`joblib`), prediction CLI, optional Streamlit app. | `predict.py`, `app.py` |
| Deep learning (future work in paper) | Optional DistilBERT fine-tuning script for comparison. | `transformer_baseline.py` |
| Quality | Unit tests for preprocessing, lexicon, NB-SVM, stats helpers, and every model. | `tests/` |

## C. What has and has not been verified

* Verified here: all code runs end-to-end, 8/8 tests pass, all CLI flags (`--demo`, `--tune`, `--no-lexicon`, `--only`) work on **synthetic demo data**.
* **Not verified here:** accuracy on the real IMDb data (no internet in the build environment). Synthetic-data numbers are meaningless; run on the real dataset to get real results. Expect the improved models to land well above the paper's ~71 %, but treat that as an expectation until you run it.
* `app.py` and `transformer_baseline.py` were only syntax-checked (Streamlit / PyTorch not installed here).

## D. Suggested next steps / further research

1. Run the three-way comparison on the full 50k set and on the paper's 10k regime; report CI + McNemar.
2. Real 3-class experiment on a dataset that has neutral labels (e.g. SST-5, Amazon 1-5 stars).
3. Domain-transfer test (train on IMDb, test on Rotten Tomatoes / Amazon) - the paper names domain shift as a challenge but never tests it.
4. Sarcasm / irony subset analysis from the error CSV.
5. Fine-tuned transformer (DistilBERT/RoBERTa) and distillation for cheap deployment.
