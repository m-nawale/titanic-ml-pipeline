import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import GradientBoostingClassifier
from features import add_basic_features

TRAIN = "data/raw/train.csv"
TEST  = "data/raw/test.csv"
OUT   = "submission_gb.csv"
SEED  = 42

def build_pipeline():
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
    return Pipeline([("prep", prep), ("clf", gb)])

def main():
    train = add_basic_features(pd.read_csv(TRAIN))
    test  = add_basic_features(pd.read_csv(TEST))

    feats = ["Pclass","Sex","Age","SibSp","Parch","Embarked",
             "Title","FamilySize","IsAlone","HasCabin","FareLog"]

    X, y = train[feats], train["Survived"]
    X_test = test[feats]

    pipe = build_pipeline()
    pipe.fit(X, y)
    preds = pipe.predict(X_test).astype(int)

    sub = pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": preds})
    sub.to_csv(OUT, index=False)
    print(f"Wrote {OUT} with shape {sub.shape}")
    print("Predicted survival rate:", sub['Survived'].mean().round(4))

if __name__ == "__main__":
    main()
