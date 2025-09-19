import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from lightgbm import LGBMClassifier
from features import add_basic_features

TRAIN = "data/raw/train.csv"
SEED = 42

def main():
    df = add_basic_features(pd.read_csv(TRAIN))
    feats = ["Pclass","Sex","Age","SibSp","Parch","Embarked",
             "Title","FamilySize","IsAlone","HasCabin","FareLog"]
    X, y = df[feats], df["Survived"]

    num = ["Age","SibSp","Parch","FamilySize","IsAlone","FareLog"]
    cat = ["Pclass","Sex","Embarked","Title","HasCabin"]

    prep = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), num),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat),
    ])

    # Keep it quiet & stable
    lgbm = LGBMClassifier(
        random_state=SEED,
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        max_depth=-1,
        n_jobs=-1,          # LightGBM does its own threading
        verbosity=-1,       # silence logs
        force_row_wise=True # avoid the auto-choosing message
    )

    pipe = Pipeline([("prep", prep), ("clf", lgbm)])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    # IMPORTANT: avoid joblib multiprocessing here to prevent cleanup issues in Docker
    scores = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy", n_jobs=1)

    print(f"LGBM 5-fold CV accuracy: {scores.mean():.4f} ± {scores.std():.4f}")
    print("Fold scores:", [round(float(s), 4) for s in scores])

if __name__ == "__main__":
    main()
