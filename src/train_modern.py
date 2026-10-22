"""
Compares MODERN models against the current best, using the exact same cross-validation folds
as train_all.py, so the numbers are directly comparable. Also creates a SHAP explanation chart.

Run:  python src\\train_modern.py
Output: reports\\modern_comparison.csv and reports\\figures\\shap_summary.png
"""
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import (HistGradientBoostingClassifier, RandomForestClassifier,
                              StackingClassifier, VotingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from pipeline_utils import encode_placement

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
SEED = 42
T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:5.0f}s] {msg}", flush=True)


def scaled(est):
    return Pipeline([("scale", StandardScaler()), ("clf", est)])


def build_models():
    lr = scaled(LogisticRegression(max_iter=2000, class_weight="balanced"))
    svm = scaled(SVC(probability=True, class_weight="balanced", random_state=SEED))
    rf = RandomForestClassifier(n_estimators=200, min_samples_leaf=2, class_weight="balanced",
                                random_state=SEED, n_jobs=1)
    xgb = XGBClassifier(n_estimators=150, max_depth=3, learning_rate=0.05,
                        eval_metric="logloss", random_state=SEED, n_jobs=1)
    cat = CatBoostClassifier(iterations=300, depth=4, learning_rate=0.05, verbose=0,
                             random_seed=SEED, thread_count=1, auto_class_weights="Balanced")
    return {
        "Logistic Regression (baseline)": lr,
        "Voting Ensemble (previous best)": VotingClassifier(
            voting="soft", estimators=[("lr", lr), ("svm", svm), ("rf", rf), ("xgb", xgb)]),
        "CatBoost": cat,
        "LightGBM": LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=8,
                                   min_child_samples=10, subsample=0.8, subsample_freq=1,
                                   colsample_bytree=0.8, class_weight="balanced",
                                   random_state=SEED, verbose=-1, n_jobs=1),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.05, max_iter=200, class_weight="balanced",
            random_state=SEED),
        "Neural Network (MLP)": scaled(MLPClassifier(
            hidden_layer_sizes=(32, 16), alpha=0.01, max_iter=3000, random_state=SEED)),
        "Stacking Ensemble": StackingClassifier(
            estimators=[("lr", lr), ("svm", svm), ("rf", rf), ("xgb", xgb), ("cat", cat)],
            final_estimator=LogisticRegression(max_iter=2000), stack_method="predict_proba",
            cv=5, n_jobs=1),
    }


def main():
    raw = pd.read_csv(ROOT / "data" / "raw" / "placement_data.csv")
    y = raw["status"].map({"Placed": 1, "Not Placed": 0}).astype(int)
    X = encode_placement(raw)
    log(f"Loaded {len(X)} students, {X.shape[1]} features")

    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=SEED)  # same folds as train_all.py
    scoring = {"accuracy": "accuracy", "f1": "f1", "roc_auc": "roc_auc"}
    models = build_models()
    rows = []
    for name, est in models.items():
        log(f"Cross-validating {name}")
        res = cross_validate(est, X, y, cv=cv, scoring=scoring, n_jobs=-1)
        rows.append({"model": name,
                     "accuracy": round(res["test_accuracy"].mean(), 4),
                     "accuracy_std": round(res["test_accuracy"].std(), 4),
                     "f1": round(res["test_f1"].mean(), 4),
                     "roc_auc": round(res["test_roc_auc"].mean(), 4)})
        log(f"    accuracy {rows[-1]['accuracy']:.3f} | F1 {rows[-1]['f1']:.3f} | AUC {rows[-1]['roc_auc']:.3f}")

    table = pd.DataFrame(rows).sort_values("roc_auc", ascending=False)
    (ROOT / "reports").mkdir(exist_ok=True)
    table.to_csv(ROOT / "reports" / "modern_comparison.csv", index=False)
    print("\nMODERN MODELS vs PREVIOUS BEST")
    print(table.to_string(index=False))

    # SHAP explanation (modern explainable-AI technique) using CatBoost
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import shap
        log("Creating SHAP explanation chart")
        cat = CatBoostClassifier(iterations=300, depth=4, learning_rate=0.05, verbose=0,
                                 random_seed=SEED, thread_count=1, auto_class_weights="Balanced")
        cat.fit(X, y)
        values = shap.TreeExplainer(cat).shap_values(X)
        out = ROOT / "reports" / "figures"
        out.mkdir(parents=True, exist_ok=True)
        plt.figure()
        shap.summary_plot(values, X, show=False, max_display=12)
        plt.tight_layout()
        plt.savefig(out / "shap_summary.png", dpi=160)
        plt.close()
        log("Saved reports\\figures\\shap_summary.png")
    except Exception as e:
        log(f"SHAP chart skipped: {e}")

    log("DONE")


if __name__ == "__main__":
    main()