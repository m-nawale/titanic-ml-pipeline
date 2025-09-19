import pandas as pd
import joblib  # ← add this import
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from features import add_basic_features

TRAIN = "data/raw/train.csv"
TEST  = "data/raw/test.csv"
OUT   = "submission_logreg_fe.csv"
MODEL = "titanic_model.pkl"   # ← add
SEED  = 42

def build_pipeline():
    num_cols = ["Age","SibSp","Parch","Fare","FamilySize","IsAlone"]
    cat_cols = ["Pclass","Sex","Embarked","Title"]

    numeric = Pipeline([("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler())])
    categorical = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                            ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    prep = ColumnTransformer([("num", numeric, num_cols),
                              ("cat", categorical, cat_cols)])

    model = LogisticRegression(max_iter=2000, C=1.0, random_state=SEED)  # C from tuning
    return Pipeline([("prep", prep), ("clf", model)])

def main():
    train = add_basic_features(pd.read_csv(TRAIN))
    test  = add_basic_features(pd.read_csv(TEST))

    features = ["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked",
                "Title","FamilySize","IsAlone"]

    X, y   = train[features], train["Survived"]
    X_test = test[features]

    pipe = build_pipeline()
    pipe.fit(X, y)
    # save model
    joblib.dump(pipe, MODEL)
    print(f"Saved model to {MODEL}")

    preds = pipe.predict(X_test).astype(int)
    sub = pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": preds})
    sub.to_csv(OUT, index=False)
    print(f"Wrote {OUT} with shape {sub.shape}")
    print("Predicted survival rate:", sub["Survived"].mean().round(4))

if __name__ == "__main__":
    main()
