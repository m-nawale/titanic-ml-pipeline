import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from features import add_basic_features

TRAIN = "data/raw/train.csv"
SEED = 42

def make_pipe(C=1.0):
    num_cols = ["Age","SibSp","Parch","Fare","FamilySize","IsAlone"]
    cat_cols = ["Pclass","Sex","Embarked","Title"]

    numeric = Pipeline([("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler())])
    categorical = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                            ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    prep = ColumnTransformer([("num", numeric, num_cols),
                              ("cat", categorical, cat_cols)])
    clf = LogisticRegression(max_iter=2000, C=C, random_state=SEED)
    return Pipeline([("prep", prep), ("clf", clf)])

def main():
    df = add_basic_features(pd.read_csv(TRAIN))
    X = df[["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked",
            "Title","FamilySize","IsAlone"]]
    y = df["Survived"]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    Cs = np.logspace(-2, 2, 9)  # 0.01 ... 100
    results = []
    for C in Cs:
        pipe = make_pipe(C=C)
        scores = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
        results.append((C, scores.mean(), scores.std()))
        print(f"C={C:.3g} -> {scores.mean():.4f} ± {scores.std():.4f}")
    best = max(results, key=lambda t: t[1])
    print("\nBest:", f"C={best[0]:.3g}", f"CV={best[1]:.4f} ± {best[2]:.4f}")

if __name__ == "__main__":
    main()
