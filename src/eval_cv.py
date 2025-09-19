import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, brier_score_loss
)
from features import add_basic_features

SEED = 42
TRAIN = "data/raw/train.csv"

def build_pipe():
    num = ["Age","SibSp","Parch","FamilySize","IsAlone","FareLog"]
    cat = ["Pclass","Sex","Embarked","Title","HasCabin"]
    prep = ColumnTransformer([
        ("num", Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
        ]), num),
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("oh", OneHotEncoder(handle_unknown="ignore")),
        ]), cat),
    ])
    clf = LogisticRegression(max_iter=2000, C=1.0, random_state=SEED)
    return Pipeline([("prep", prep), ("clf", clf)]), num+cat

def main():
    df = add_basic_features(pd.read_csv(TRAIN))
    feats = ["Pclass","Sex","Age","SibSp","Parch","Embarked",
             "Title","FamilySize","IsAlone","HasCabin","FareLog"]
    X, y = df[feats], df["Survived"]

    pipe, _ = build_pipe()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    # Out-of-fold predicted probabilities for class 1
    p = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba", n_jobs=-1)[:,1]
    y_pred = (p >= 0.5).astype(int)

    print("ROC-AUC:", round(roc_auc_score(y, p), 4))
    print("PR-AUC :", round(average_precision_score(y, p), 4))
    print("Brier  :", round(brier_score_loss(y, p), 4))
    print("\nConfusion @0.5 threshold:")
    print(confusion_matrix(y, y_pred))
    print("\nClassification report:")
    print(classification_report(y, y_pred, digits=3))

if __name__ == "__main__":
    main()
