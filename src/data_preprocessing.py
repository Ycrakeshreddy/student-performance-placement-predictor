import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

RAW_PATH = "data/raw/placement_data.csv"
CLEAN_PATH = "data/processed/cleaned_data.csv"

NUMERIC_COLS = ["ssc_p", "hsc_p", "degree_p", "etest_p", "mba_p"]
CATEGORICAL_COLS = ["gender", "ssc_b", "hsc_b", "hsc_s", "degree_t",
                    "workex", "specialisation"]


def load_and_clean():
    df = pd.read_csv(RAW_PATH)

    # Drop columns that are not useful or would leak the answer
    df = df.drop(columns=["sl_no", "salary"])

    # Target: Placed = 1, Not Placed = 0
    df["status"] = df["status"].map({"Placed": 1, "Not Placed": 0})

    # One-hot encode categorical columns
    df = pd.get_dummies(df, columns=CATEGORICAL_COLS, drop_first=True, dtype=int)

    df.to_csv(CLEAN_PATH, index=False)
    return df


def prepare_data():
    df = load_and_clean()
    X = df.drop(columns=["status"])
    y = df["status"]

    # Split FIRST, then fit the scaler on training data only (avoids leakage)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train[NUMERIC_COLS] = scaler.fit_transform(X_train[NUMERIC_COLS])
    X_test[NUMERIC_COLS] = scaler.transform(X_test[NUMERIC_COLS])

    joblib.dump(scaler, "models/scaler.pkl")
    joblib.dump(list(X.columns), "models/feature_columns.pkl")
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = prepare_data()
    print("Train shape:", X_train.shape)
    print("Test shape:", X_test.shape)
    print("Placed ratio in train:", round(y_train.mean(), 3))
    print("Features:", list(X_train.columns))