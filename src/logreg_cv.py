import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from features import add_basic_features

TRAIN = "data/raw/train.csv"
SEED = 42

def main():
    df = pd.read_csv(TRAIN)
    df = add_basic_features(df)  # ← ensure engineered cols exist

    target = "Survived"
    features = [
        "Pclass","Sex","Age","SibSp","Parch","Embarked",
        "Title","FamilySize","IsAlone","HasCabin","FareLog"  # ← NEW
    ]
    # quick sanity print so we know we’re using the right set
    print("Using features:", features)

    X = df[features].copy()
    y = df[target].copy()

    num_cols = ["Age","SibSp","Parch","FamilySize","IsAlone","FareLog"]  # ← FareLog
    cat_cols = ["Pclass","Sex","Embarked","Title","HasCabin"]            # ← HasCabin

    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    prep = ColumnTransformer([
        ("num", numeric, num_cols),
        ("cat", categorical, cat_cols),
    ])

    model = LogisticRegression(max_iter=2000, C=1.0, random_state=SEED)

    pipe = Pipeline([
        ("prep", prep),
        ("clf", model),
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    scores = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy", n_jobs=-1)

    print(f"LogReg+FE (HasCabin+FareLog) 5-fold CV: {scores.mean():.4f} ± {scores.std():.4f}")
    print("Fold scores:", [round(float(s), 4) for s in scores])

if __name__ == "__main__":
    main()
