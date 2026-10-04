"""End-to-end experiment:  python -m msa.train --data "data/IMDB Dataset.csv"   (or --demo)"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score

from . import evaluate as ev
from .data import load_any, make_demo_data, split_data
from .models import model_zoo
from .preprocessing import prepare_frame


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", help="Kaggle CSV, generic CSV, or aclImdb folder")
    ap.add_argument("--demo", action="store_true", help="use synthetic data (smoke test)")
    ap.add_argument("--demo-size", type=int, default=3000)
    ap.add_argument("--sample", type=int, default=None, help="sub-sample N reviews (10000 = paper's regime)")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--cv", type=int, default=3, help="CV folds on train set for model selection (0 = off)")
    ap.add_argument("--tune", action="store_true", help="grid-search hyper-parameters (slower)")
    ap.add_argument("--no-lexicon", action="store_true", help="ablation: drop lexicon features")
    ap.add_argument("--only", nargs="*", help="substring filter on model names")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args(argv)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.demo:
        df = make_demo_data(a.demo_size, a.seed)
        print("!! DEMO MODE: synthetic data, results are only a pipeline smoke test.")
    elif a.data:
        df = load_any(a.data)
    else:
        ap.error("give --data PATH or --demo")

    train_raw, test_raw = split_data(df, a.test_size, a.seed, a.sample)
    print(f"train={len(train_raw)} test={len(test_raw)} classes={sorted(train_raw.label.unique())}")
    t0 = time.time()
    Xtr, Xte = prepare_frame(train_raw.review), prepare_frame(test_raw.review)
    ytr, yte = train_raw["label"].to_numpy(), test_raw["label"].to_numpy()
    print(f"preprocessing: {time.time() - t0:.1f}s")

    binary = len(set(ytr)) == 2
    zoo = model_zoo(len(Xtr), a.seed, not a.no_lexicon, binary)
    if a.only:
        zoo = {k: v for k, v in zoo.items() if any(s.lower() in k.lower() for s in a.only)}

    results, rows, fitted = {}, [], {}
    skf = StratifiedKFold(a.cv if a.cv >= 2 else 2, shuffle=True, random_state=a.seed)
    for name, (pipe, grid) in zoo.items():
        t = time.time()
        row = {"model": name}
        if a.tune and grid:
            gs = GridSearchCV(pipe, grid, cv=skf, scoring="accuracy", n_jobs=1).fit(Xtr, ytr)
            pipe, row["best_params"] = gs.best_estimator_, str(gs.best_params_)
        else:
            pipe.fit(Xtr, ytr)
        row["train_sec"] = round(time.time() - t, 1)
        if a.cv >= 2 and not a.tune:
            sc = cross_val_score(pipe, Xtr, ytr, cv=skf, scoring="accuracy")
            row["cv_acc_mean"], row["cv_acc_std"] = sc.mean(), sc.std()
        elif a.tune:
            row["cv_acc_mean"], row["cv_acc_std"] = gs.best_score_, np.nan
        pred = pipe.predict(Xte)
        proba = pipe.predict_proba(Xte) if hasattr(pipe, "predict_proba") else None
        score = (proba[:, list(pipe.classes_).index("positive")] if proba is not None and binary
                 else (pipe.decision_function(Xte) if hasattr(pipe, "decision_function") and binary else None))
        m = ev.compute_metrics(yte, pred, proba, pipe.classes_)
        lo, hi = ev.bootstrap_ci((pred == yte).astype(float), seed=a.seed)
        row.update(m, acc_ci_low=lo, acc_ci_high=hi)
        rows.append(row)
        results[name] = {"y_true": yte, "y_pred": pred, "proba": proba, "score": score}
        fitted[name] = pipe
        print(f"{name:34s} acc={m['accuracy']:.4f} f1={m['f1_macro']:.4f} ({row['train_sec']}s)")

    res = pd.DataFrame(rows).set_index("model")
    res.round(4).to_csv(out / "results.csv")

    # significance: every model vs. best paper replica
    paper = [n for n in res.index if n.startswith("Paper")]
    if paper:
        ref = res.loc[paper, "accuracy"].idxmax()
        ref_ok = results[ref]["y_pred"] == yte
        pv = {n: ev.mcnemar_exact(results[n]["y_pred"] == yte, ref_ok) for n in res.index if n != ref}
        pd.Series(pv, name=f"mcnemar_p_vs[{ref}]").round(5).to_csv(out / "mcnemar.csv")
    # model selection by CV (never by the test set), falls back to test only if CV is off
    key = "cv_acc_mean" if "cv_acc_mean" in res and res["cv_acc_mean"].notna().any() else "accuracy"
    cand = res[res.index.str.startswith("Improved")] if any(res.index.str.startswith("Improved")) else res
    best = cand[key].idxmax()
    print(f"\nBest model by {key}: {best}")

    ev.plot_confusions(results, sorted(set(ytr)), out / "confusion_matrices.png")
    if binary:
        ev.plot_roc(results, out / "roc_curves.png")
    ev.plot_comparison(res, out / "model_comparison.png")
    for n in (best, "Improved: Logistic Reg."):
        if n in fitted and (tf := ev.top_features(fitted[n])) is not None:
            tf.to_csv(out / "top_features.csv", index=False)
            break
    ev.error_analysis(test_raw, results[best]["y_pred"], results[best]["proba"], fitted[best].classes_).to_csv(
        out / "error_analysis.csv", index=False)
    joblib.dump(fitted[best], out / "best_model.joblib")

    cols = ["accuracy", "precision_macro", "recall_macro", "f1_macro", "roc_auc", "mcc", "acc_ci_low", "acc_ci_high"]
    with open(out / "report.md", "w") as f:
        f.write("# Results\n\n" + ("**DEMO (synthetic) data - not meaningful.**\n\n" if a.demo else ""))
        f.write(f"train={len(Xtr)}, test={len(Xte)}, seed={a.seed}, lexicon={'off' if a.no_lexicon else 'on'}\n\n")
        f.write(res[[c for c in cols if c in res]].round(4).to_markdown() + f"\n\nSelected model: **{best}**\n")
    print(f"\nArtifacts written to {out.resolve()}")
    return res


if __name__ == "__main__":
    main()
