import json
import os

import joblib
import pandas as pd

from pipeline_utils import CAT, NUM, encode_placement

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
M = {}
LOAD_ERROR = "Models are not trained yet. Open the Training tab and press Train models."


def load():
    """Load (or reload) all saved models. Called at start and after retraining."""
    global LOAD_ERROR
    M["model"] = joblib.load(os.path.join(BASE, "placement_model.pkl"))
    M["cols"] = joblib.load(os.path.join(BASE, "feature_columns.pkl"))
    for tag, suffix in [("early", ""), ("mid", "_mid")]:
        M[tag] = {
            "model": joblib.load(os.path.join(BASE, f"performance_model{suffix}.pkl")),
            "cols": joblib.load(os.path.join(BASE, f"performance_columns{suffix}.pkl")),
            "defaults": joblib.load(os.path.join(BASE, f"performance_defaults{suffix}.pkl")),
        }
    try:
        with open(os.path.join(BASE, "metrics.json")) as f:
            m = json.load(f)
        M["mae"] = {"early": _best_mae(m["performance_early"]), "mid": _best_mae(m["performance_mid"])}
    except Exception:
        M["mae"] = {"early": None, "mid": None}
    LOAD_ERROR = None


def _best_mae(block):
    for row in block["table"]:
        if row["model"] == block["best"]:
            return row["mae"]
    return None


def _prob(d):
    row = pd.DataFrame([{c: d[c] for c in NUM + CAT}])
    X = encode_placement(row).reindex(columns=M["cols"], fill_value=0)
    return float(M["model"].predict_proba(X)[0][1])


def predict_placement(d):
    base = _prob(d)
    scenarios = []

    def add(text, change):
        e = dict(d)
        change(e)
        p = _prob(e)
        scenarios.append({"text": text, "prob": round(p * 100, 1), "delta": round((p - base) * 100, 1)})

    if d["workex"] == "No":
        add("Gain work experience (internship)", lambda e: e.update(workex="Yes"))
    if float(d["etest_p"]) < 95:
        add("Raise employability test by 10 points", lambda e: e.update(etest_p=min(100.0, float(d["etest_p"]) + 10)))
    if float(d["degree_p"]) < 95:
        add("Raise degree percentage by 5 points", lambda e: e.update(degree_p=min(100.0, float(d["degree_p"]) + 5)))
    if float(d["mba_p"]) < 95:
        add("Raise MBA percentage by 5 points", lambda e: e.update(mba_p=min(100.0, float(d["mba_p"]) + 5)))
    scenarios.sort(key=lambda s: -s["delta"])

    prob = base * 100
    if prob >= 70:
        label, band = "High chance of placement", "high"
    elif prob >= 40:
        label, band = "Moderate chance, room to improve", "mid"
    else:
        label, band = "Needs extra support", "low"
    return {"prob": round(prob, 1), "label": label, "band": band, "scenarios": scenarios}


def predict_performance(d):
    use_mid = d.get("g1") not in (None, "") and d.get("g2") not in (None, "")
    tag = "mid" if use_mid else "early"
    blk = M[tag]
    row = {c: float(blk["defaults"].get(c, 0)) for c in blk["cols"]}
    for c in ["age", "Medu", "Fedu", "studytime", "failures", "absences", "goout", "health"]:
        if c in row and d.get(c) not in (None, ""):
            row[c] = float(d[c])
    row["sex_M"] = 1.0 if d.get("sex") == "M" else 0.0
    for c in ["higher", "internet", "schoolsup"]:
        key = f"{c}_yes"
        if key in row:
            row[key] = 1.0 if d.get(c) == "yes" else 0.0
    if use_mid:
        row["G1"], row["G2"] = float(d["g1"]), float(d["g2"])
    X = pd.DataFrame([row])[blk["cols"]]
    score = float(blk["model"].predict(X)[0])
    return {"score": round(max(0.0, min(20.0, score)), 1), "out_of": 20, "mode": tag,
            "mae": M["mae"].get(tag)}


def run(d):
    if LOAD_ERROR:
        raise RuntimeError(LOAD_ERROR)
    return {"placement": predict_placement(d), "performance": predict_performance(d)}


try:
    load()
except Exception:
    pass  # app still starts; train from the Training tab