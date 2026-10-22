import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)
from xgboost import XGBClassifier

from data_preprocessing import prepare_data

X_train, X_test, y_train, y_test = prepare_data()

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42,
                                            class_weight="balanced"),
    "XGBoost": XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05,
                             eval_metric="logloss", random_state=42),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
rows = []

for name, model in models.items():
    cv_f1 = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1").mean()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]
    rows.append({
        "Model": name,
        "CV F1": round(cv_f1, 3),
        "Accuracy": round(accuracy_score(y_test, pred), 3),
        "Precision": round(precision_score(y_test, pred), 3),
        "Recall": round(recall_score(y_test, pred), 3),
        "F1": round(f1_score(y_test, pred), 3),
        "ROC-AUC": round(roc_auc_score(y_test, prob), 3),
    })

results = pd.DataFrame(rows).sort_values("CV F1", ascending=False)
print("\n===== MODEL COMPARISON =====")
print(results.to_string(index=False))
results.to_csv("reports/model_comparison.csv", index=False)

# Tune Random Forest with GridSearchCV
grid = GridSearchCV(
    RandomForestClassifier(random_state=42, class_weight="balanced"),
    {"n_estimators": [100, 200, 300], "max_depth": [3, 5, 8, None],
     "min_samples_leaf": [1, 2, 4]},
    cv=cv, scoring="f1", n_jobs=-1,
)
grid.fit(X_train, y_train)
best = grid.best_estimator_
print("\nBest RF params:", grid.best_params_)
print("Best CV F1:", round(grid.best_score_, 3))

pred = best.predict(X_test)
print("\n===== TUNED RANDOM FOREST ON TEST SET =====")
print("Accuracy:", round(accuracy_score(y_test, pred), 3))
print("F1:", round(f1_score(y_test, pred), 3))
print("Confusion matrix:\n", confusion_matrix(y_test, pred))

# Top features
imp = pd.Series(best.feature_importances_, index=X_train.columns)
print("\nTop 5 features:\n", imp.sort_values(ascending=False).head(5).round(3))

joblib.dump(best, "models/placement_model.pkl")
print("\nSaved: models/placement_model.pkl")