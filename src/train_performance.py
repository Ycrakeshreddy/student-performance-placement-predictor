import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------- 1. Load and clean ----------
df = pd.read_csv("data/raw/student_data.csv", sep=";")
df.columns = df.columns.str.strip()
df = df.dropna(how="all").reset_index(drop=True)
df = df.dropna()

if len(df) < 100:
    raise SystemExit("Dataset looks damaged: only %d rows left. "
                     "Re-run src\\fix_student_data.py." % len(df))
print("Shape:", df.shape)

# ---------- 2. Drop G1 and G2 ----------
# G1 and G2 are earlier grades that almost give away G3,
# so we predict from habits and background instead.
df = df.drop(columns=["G1", "G2"])

# ---------- 3. Convert numbers, then encode text columns ----------
for c in df.columns:
    try:
        df[c] = pd.to_numeric(df[c])
    except (ValueError, TypeError):
        pass

text_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
print("Text columns encoded:", text_cols)
df = pd.get_dummies(df, columns=text_cols, drop_first=True, dtype=int)

X = df.drop(columns=["G3"])
y = df["G3"]

# ---------- 4. Split ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------- 5. Train and compare ----------
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest Regressor": RandomForestRegressor(
        n_estimators=300, max_depth=8, min_samples_leaf=3, random_state=42),
}

cv = KFold(n_splits=5, shuffle=True, random_state=42)
rows = []
trained = {}

for name, model in models.items():
    cv_r2 = cross_val_score(model, X_train, y_train, cv=cv, scoring="r2").mean()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    trained[name] = model
    rows.append({
        "Model": name,
        "CV R2": round(cv_r2, 3),
        "MAE": round(mean_absolute_error(y_test, pred), 3),
        "RMSE": round(float(np.sqrt(mean_squared_error(y_test, pred))), 3),
        "R2": round(r2_score(y_test, pred), 3),
    })

results = pd.DataFrame(rows).sort_values("CV R2", ascending=False)
print("\n===== REGRESSION COMPARISON =====")
print(results.to_string(index=False))
results.to_csv("reports/regression_comparison.csv", index=False)

# ---------- 6. Pick the best and save ----------
best_name = results.iloc[0]["Model"]
best = trained[best_name]
print("\nBest model:", best_name)

if hasattr(best, "feature_importances_"):
    imp = pd.Series(best.feature_importances_, index=X.columns)
    print("\nTop 5 features:\n", imp.sort_values(ascending=False).head(5).round(3))

joblib.dump(best, "models/performance_model.pkl")
joblib.dump(list(X.columns), "models/performance_columns.pkl")
joblib.dump(X.median().to_dict(), "models/performance_defaults.pkl")
print("\nSaved: models/performance_model.pkl, performance_columns.pkl, performance_defaults.pkl")
print("Feature columns:", list(X.columns))