import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder  # trees don't need scaling
from sklearn.ensemble import GradientBoostingClassifier
from features import add_basic_features

TRAIN = "data/raw/train.csv"
SEED = 42

def main():
    df = add_basic_features(pd.read_csv(TRAIN))

    features = ["Pclass","Sex","Age","SibSp","Parch","Embarked",
                "Title","FamilySize","IsAlone","HasCabin","FareLog"]
    target = "Survived"

    X, y = df[features], df[target]

    num_cols = ["Age","SibSp","Parch","FamilySize","IsAlone","FareLog"]
    cat_cols = ["Pclass","Sex","Embarked","Title","HasCabin"]

    prep = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), num_cols),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), cat_cols),
    ])

    gb = GradientBoostingClassifier(
        random_state=SEED,
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
    )

    pipe = Pipeline([("prep", prep), ("clf", gb)])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    scores = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
    print(f"GB 5-fold CV accuracy: {scores.mean():.4f} ± {scores.std():.4f}")
    print("Fold scores:", [round(float(s), 4) for s in scores])

if __name__ == "__main__":
    main()
