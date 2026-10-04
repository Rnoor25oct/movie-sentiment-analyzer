# Results

**DEMO (synthetic) data - not meaningful.**

train=1600, test=400, seed=42, lexicon=on

| model                          |   accuracy |   precision_macro |   recall_macro |   f1_macro |   roc_auc |    mcc |   acc_ci_low |   acc_ci_high |
|:-------------------------------|-----------:|------------------:|---------------:|-----------:|----------:|-------:|-------------:|--------------:|
| Paper: Naive Bayes (BoW)       |     0.9125 |            0.913  |         0.9125 |     0.9125 |    0.9096 | 0.8255 |       0.885  |        0.94   |
| Paper: Logistic Reg. (BoW)     |     0.9    |            0.9004 |         0.9    |     0.9    |    0.91   | 0.8004 |       0.87   |        0.9275 |
| Paper: Linear SVM (BoW)        |     0.8675 |            0.868  |         0.8675 |     0.8675 |  nan      | 0.7355 |       0.835  |        0.9    |
| Improved: Logistic Reg.        |     0.9225 |            0.9234 |         0.9225 |     0.9225 |    0.9227 | 0.8459 |       0.895  |        0.9475 |
| Improved: Linear SVM           |     0.9225 |            0.923  |         0.9225 |     0.9225 |    0.9229 | 0.8455 |       0.895  |        0.9475 |
| Improved: Naive Bayes          |     0.92   |            0.9204 |         0.92   |     0.92   |    0.9144 | 0.8404 |       0.8925 |        0.9475 |
| Improved: NB-SVM               |     0.9225 |            0.923  |         0.9225 |     0.9225 |    0.9156 | 0.8455 |       0.895  |        0.9475 |
| Improved: Soft-voting Ensemble |     0.9225 |            0.923  |         0.9225 |     0.9225 |    0.9201 | 0.8455 |       0.895  |        0.9475 |

Selected model: **Improved: Soft-voting Ensemble**
