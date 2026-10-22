"""Shared helpers so training and prediction build features in exactly the same way."""
import pandas as pd

NUM = ["ssc_p", "hsc_p", "degree_p", "etest_p", "mba_p"]
CAT = ["gender", "ssc_b", "hsc_b", "hsc_s", "degree_t", "workex", "specialisation"]

PRETTY = {
    "ssc_p": "SSC % (10th)",
    "hsc_p": "HSC % (12th)",
    "degree_p": "Degree %",
    "etest_p": "Employability test %",
    "mba_p": "MBA %",
    "avg_academic": "Average academic %",
    "degree_vs_school": "Degree vs school gap",
    "workex_Yes": "Work experience: Yes",
    "workex_No": "Work experience: No",
    "specialisation_Mkt&HR": "Specialisation: Mkt&HR",
    "specialisation_Mkt&Fin": "Specialisation: Mkt&Fin",
    "gender_M": "Gender: Male",
    "gender_F": "Gender: Female",
    "absences": "Absences",
    "failures": "Past failures",
    "studytime": "Study time",
    "goout": "Going out",
    "health": "Health",
    "age": "Age",
    "Medu": "Mother's education",
    "Fedu": "Father's education",
    "G1": "Period 1 grade (G1)",
    "G2": "Period 2 grade (G2)",
    "higher_yes": "Wants higher education",
    "internet_yes": "Internet at home",
    "schoolsup_yes": "Extra school support",
    "sex_M": "Sex: Male",
}

PREFIX = {
    "degree_t": "Degree type", "hsc_s": "HSC stream", "ssc_b": "SSC board", "hsc_b": "HSC board",
    "gender": "Gender", "workex": "Work experience", "specialisation": "Specialisation",
    "Mjob": "Mother's job", "Fjob": "Father's job", "reason": "School choice reason",
    "guardian": "Guardian", "school": "School", "address": "Address", "famsize": "Family size",
    "Pstatus": "Parents status", "famsup": "Family support", "paid": "Paid classes",
    "activities": "Activities", "nursery": "Nursery", "romantic": "Relationship",
}


def pretty(name):
    if name in PRETTY:
        return PRETTY[name]
    for key, nice in PREFIX.items():
        if name.startswith(key + "_"):
            return nice + ": " + name[len(key) + 1:]
    return name


def encode_placement(df):
    """Raw placement rows -> numeric feature table (with two engineered features)."""
    X = df[NUM + CAT].copy()
    for c in NUM:
        X[c] = pd.to_numeric(X[c])
    X["avg_academic"] = X[["ssc_p", "hsc_p", "degree_p"]].mean(axis=1)
    X["degree_vs_school"] = X["degree_p"] - (X["ssc_p"] + X["hsc_p"]) / 2
    X = pd.get_dummies(X, columns=CAT, dtype=int)
    return X