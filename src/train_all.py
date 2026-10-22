"""
Trains every model for the project and saves:
  - models/*.pkl      (models used by the web app)
  - models/metrics.json  (numbers and chart data shown in the web app)
  - reports/*.csv     (tables for your report)

Run from the project folder:   python src\\train_all.py
"""
import json
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import (ExtraTreesRegressor, GradientBoostingClassifier,
                              GradientBoostingRegressor, RandomForestClassifier,
                              RandomForestRegressor, VotingClassifier)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import (KFold, RandomizedSearchCV, RepeatedKFold,
                                     RepeatedStratifiedKFold, StratifiedKFold,
                                     cross_val_predict, cross_validate,
                                     learning_curve, train_test_split)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier, XGBRegressor

from pipeline_utils import encode_placement, pretty

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
MODELS.mkdir(exist_ok=True)
REPORTS.mkdir(exist_ok=True)
SEED = 42
T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:5.0f}s] {msg}", flush=True)


def scaled(est):
    return Pipeline([("scale", StandardScaler()), ("clf", est)])


def plain(est):
    """If a search object was fitted, return the best underlying estimator."""
    return est.best_estimator_ if hasattr(est, "best_estimator_") else est


def r(x, d=4):
    return round(float(x), d)


# PART 1: PLACEMENT (classification)
def train_placement():
    log("PLACEMENT: loading data")
    raw = pd.read_csv(ROOT / "data" / "raw" / "placement_data.csv")
    y = raw["status"].map({"Placed": 1, "Not Placed": 0}).astype(int)
    X = encode_placement(raw)           # salary and sl_no are never used (no leakage)
    cols = list(X.columns)
    log(f"PLACEMENT: {X.shape[0]} students, {X.shape[1]} features, placed rate {y.mean():.1%}")

    def tuned(est, grid):
        return RandomizedSearchCV(est, grid, n_iter=6, cv=3, scoring="f1",
                                  random_state=SEED, n_jobs=1)

    rf = scaled(RandomForestClassifier(class_weight="balanced", random_state=SEED, n_jobs=1))
    gb = scaled(GradientBoostingClassifier(random_state=SEED))
    xgb = scaled(XGBClassifier(eval_metric="logloss", random_state=SEED, n_jobs=1,
                               tree_method="hist"))

    candidates = {
        "Logistic Regression": scaled(LogisticRegression(max_iter=2000, class_weight="balanced")),
        "SVM (RBF)": scaled(SVC(probability=True, class_weight="balanced", random_state=SEED)),
        "KNN": scaled(KNeighborsClassifier(n_neighbors=9)),
        "Decision Tree": scaled(DecisionTreeClassifier(max_depth=4, class_weight="balanced",
                                                       random_state=SEED)),
        "Random Forest (tuned)": tuned(rf, {
            "clf__n_estimators": [100, 200, 300], "clf__max_depth": [3, 5, 8, None],
            "clf__min_samples_leaf": [1, 2, 4], "clf__max_features": ["sqrt", 0.5]}),
        "Gradient Boosting (tuned)": tuned(gb, {
            "clf__n_estimators": [80, 150, 250], "clf__learning_rate": [0.03, 0.05, 0.1],
            "clf__max_depth": [2, 3], "clf__subsample": [0.7, 0.9, 1.0]}),
        "XGBoost (tuned)": tuned(xgb, {
            "clf__n_estimators": [100, 200, 300], "clf__max_depth": [2, 3, 4],
            "clf__learning_rate": [0.03, 0.05, 0.1], "clf__subsample": [0.7, 0.9, 1.0],
            "clf__colsample_bytree": [0.6, 0.8, 1.0]}),
        "Voting Ensemble": VotingClassifier(voting="soft", estimators=[
            ("lr", scaled(LogisticRegression(max_iter=2000, class_weight="balanced"))),
            ("svm", scaled(SVC(probability=True, class_weight="balanced", random_state=SEED))),
            ("rf", RandomForestClassifier(n_estimators=200, min_samples_leaf=2,
                                          class_weight="balanced", random_state=SEED, n_jobs=1)),
            ("xgb", XGBClassifier(n_estimators=150, max_depth=3, learning_rate=0.05,
                                  eval_metric="logloss", random_state=SEED, n_jobs=1)),
        ]),
    }

    # Repeated stratified 5-fold CV. Tuned models are tuned INSIDE each fold (nested CV),
    # so the scores below are honest and not inflated by tuning on the test data.
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=SEED)
    scoring = {"accuracy": "accuracy", "precision": "precision", "recall": "recall",
               "f1": "f1", "roc_auc": "roc_auc"}
    table = []
    for name, est in candidates.items():
        log(f"PLACEMENT: training and cross-validating {name}")
        res = cross_validate(est, X, y, cv=cv, scoring=scoring, n_jobs=-1,
                             return_train_score=True)
        table.append({
            "model": name,
            "accuracy": r(res["test_accuracy"].mean()), "accuracy_std": r(res["test_accuracy"].std()),
            "precision": r(res["test_precision"].mean()), "recall": r(res["test_recall"].mean()),
            "f1": r(res["test_f1"].mean()), "roc_auc": r(res["test_roc_auc"].mean()),
            "train_accuracy": r(res["train_accuracy"].mean()),
        })
        log(f"    accuracy {table[-1]['accuracy']:.3f} | F1 {table[-1]['f1']:.3f} | "
            f"AUC {table[-1]['roc_auc']:.3f}")

    df_table = pd.DataFrame(table).sort_values("roc_auc", ascending=False)
    df_table.to_csv(REPORTS / "model_comparison.csv", index=False)
    best_name = df_table.iloc[0]["model"]
    log(f"PLACEMENT: best model by cross-validated ROC-AUC = {best_name}")

    # Out-of-fold predictions for the best model (every student predicted while unseen)
    best_est = candidates[best_name]
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    oof = cross_val_predict(clone(best_est), X, y, cv=skf, method="predict_proba", n_jobs=-1)[:, 1]
    oof_pred = (oof >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, oof_pred).ravel()
    fpr, tpr, _ = roc_curve(y, oof)
    step = max(1, len(fpr) // 60)

    # Final model trained on all data and saved for the app
    log("PLACEMENT: fitting final model on all data")
    final = clone(best_est).fit(X, y)
    final_model = plain(final)
    joblib.dump(final_model, MODELS / "placement_model.pkl")
    joblib.dump(cols, MODELS / "feature_columns.pkl")

    log("PLACEMENT: computing feature importance")
    pi = permutation_importance(final_model, X, y, scoring="roc_auc", n_repeats=15,
                                random_state=SEED, n_jobs=-1)
    imp = sorted(zip(cols, pi.importances_mean), key=lambda t: -t[1])[:10]

    log("PLACEMENT: learning curve")
    sizes, tr, va = learning_curve(clone(final_model), X, y, cv=skf, scoring="accuracy",
                                   train_sizes=np.linspace(0.3, 1.0, 6), n_jobs=-1,
                                   random_state=SEED, shuffle=True)

    log("PLACEMENT: XGBoost training curve (loss per boosting round)")
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=SEED)
    xm = XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.9,
                       eval_metric="logloss", random_state=SEED, n_jobs=1)
    xm.fit(Xtr, ytr, eval_set=[(Xtr, ytr), (Xte, yte)], verbose=False)
    ev = xm.evals_result()
    rounds = list(range(1, 301))[::5]
    xgb_curve = {"rounds": rounds,
                 "train": [r(v) for v in ev["validation_0"]["logloss"][::5]],
                 "val": [r(v) for v in ev["validation_1"]["logloss"][::5]]}

    # Simple data insights for the "Data insights" tab
    eda = {"placement_rate": [], "avg_by_status": []}
    for col, nice in [("workex", "Work experience"), ("specialisation", "Specialisation"),
                      ("gender", "Gender"), ("degree_t", "Degree type")]:
        rate = raw.groupby(col)["status"].apply(lambda s: (s == "Placed").mean())
        eda["placement_rate"] += [{"label": f"{nice}: {k}", "value": r(v)} for k, v in rate.items()]
    for col in ["ssc_p", "hsc_p", "degree_p", "etest_p", "mba_p"]:
        g = raw.groupby("status")[col].mean()
        eda["avg_by_status"].append({"feature": pretty(col), "placed": r(g["Placed"], 2),
                                     "not_placed": r(g["Not Placed"], 2)})

    return {
        "n": int(len(X)), "n_features": int(X.shape[1]), "placed_rate": r(y.mean()),
        "best": best_name, "table": df_table.to_dict("records"),
        "confusion": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "roc": {"fpr": [r(v) for v in fpr[::step]], "tpr": [r(v) for v in tpr[::step]],
                "auc": r(roc_auc_score(y, oof))},
        "importance": [{"name": pretty(n), "value": r(v)} for n, v in imp],
        "learning_curve": {"sizes": [int(s) for s in sizes],
                           "train": [r(v) for v in tr.mean(axis=1)],
                           "val": [r(v) for v in va.mean(axis=1)]},
        "xgb_curve": xgb_curve,
    }, eda


# PART 2: MARKS (regression), two versions
def load_students():
    df = pd.read_csv(ROOT / "data" / "raw" / "student_data.csv", sep=";")
    df.columns = df.columns.str.strip()
    df = df.dropna(how="all").dropna().reset_index(drop=True)
    if len(df) < 100:
        raise SystemExit("student_data.csv looks damaged. Re-run src\\fix_student_data.py.")
    for c in df.columns:
        try:
            df[c] = pd.to_numeric(df[c])
        except (ValueError, TypeError):
            pass
    return df


def train_marks(df, use_prev_grades, tag):
    label = "MID-TERM (uses G1, G2)" if use_prev_grades else "EARLY (no previous grades)"
    log(f"MARKS {label}: preparing data")
    d = df.copy() if use_prev_grades else df.drop(columns=["G1", "G2"])
    text_cols = [c for c in d.columns if not pd.api.types.is_numeric_dtype(d[c])]
    d = pd.get_dummies(d, columns=text_cols, drop_first=True, dtype=int)
    X, y = d.drop(columns=["G3"]), d["G3"]
    cols = list(X.columns)

    candidates = {
        "Ridge Regression": scaled(Ridge(alpha=10.0)),
        "Random Forest": RandomForestRegressor(n_estimators=250, max_depth=8, min_samples_leaf=3,
                                               random_state=SEED, n_jobs=1),
        "Extra Trees": ExtraTreesRegressor(n_estimators=250, max_depth=10, min_samples_leaf=2,
                                           random_state=SEED, n_jobs=1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=250, learning_rate=0.03,
                                                       max_depth=3, subsample=0.8,
                                                       random_state=SEED),
        "XGBoost": XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.03,
                                subsample=0.8, colsample_bytree=0.8, random_state=SEED,
                                n_jobs=1, tree_method="hist"),
    }
    cv = RepeatedKFold(n_splits=5, n_repeats=2, random_state=SEED)
    scoring = {"r2": "r2", "mae": "neg_mean_absolute_error", "rmse": "neg_root_mean_squared_error"}
    table = []
    for name, est in candidates.items():
        log(f"MARKS {tag}: cross-validating {name}")
        res = cross_validate(est, X, y, cv=cv, scoring=scoring, n_jobs=-1, return_train_score=True)
        table.append({"model": name, "r2": r(res["test_r2"].mean()), "r2_std": r(res["test_r2"].std()),
                      "mae": r(-res["test_mae"].mean()), "rmse": r(-res["test_rmse"].mean()),
                      "train_r2": r(res["train_r2"].mean())})
        log(f"    R2 {table[-1]['r2']:.3f} | MAE {table[-1]['mae']:.2f} marks")

    df_table = pd.DataFrame(table).sort_values("r2", ascending=False)
    df_table.to_csv(REPORTS / f"regression_comparison_{tag}.csv", index=False)
    best_name = df_table.iloc[0]["model"]
    log(f"MARKS {tag}: best model = {best_name}")

    kf = KFold(5, shuffle=True, random_state=SEED)
    oof = cross_val_predict(clone(candidates[best_name]), X, y, cv=kf, n_jobs=-1)
    final = clone(candidates[best_name]).fit(X, y)
    joblib.dump(final, MODELS / f"performance_model{'_mid' if use_prev_grades else ''}.pkl")
    joblib.dump(cols, MODELS / f"performance_columns{'_mid' if use_prev_grades else ''}.pkl")
    joblib.dump(X.median().to_dict(),
                MODELS / f"performance_defaults{'_mid' if use_prev_grades else ''}.pkl")

    pi = permutation_importance(final, X, y, scoring="r2", n_repeats=10, random_state=SEED, n_jobs=-1)
    imp = sorted(zip(cols, pi.importances_mean), key=lambda t: -t[1])[:8]
    return {
        "n": int(len(X)), "best": best_name, "table": df_table.to_dict("records"),
        "scatter": [[int(a), r(p, 2)] for a, p in zip(y, np.clip(oof, 0, 20))][::2],
        "importance": [{"name": pretty(n), "value": r(v)} for n, v in imp],
    }


def main():
    log("Training started")
    placement, eda = train_placement()
    df = load_students()
    early = train_marks(df, False, "early")
    mid = train_marks(df, True, "mid")
    metrics = {"generated": time.strftime("%d %b %Y, %H:%M"), "placement": placement,
               "performance_early": early, "performance_mid": mid, "eda": eda}
    with open(MODELS / "metrics.json", "w") as f:
        json.dump(metrics, f)
    log("DONE. Saved models and metrics. Restart is not needed if you trained from the web app.")


if __name__ == "__main__":
    main()