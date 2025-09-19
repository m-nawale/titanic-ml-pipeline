import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder  # no scaling needed for trees
from sklearn.ensemble import RandomForestClassifier

TRAIN = "data/raw/train.csv"
SEED = 42

def main():
    df = pd.read_csv(TRAIN)

    # --- minimal feature engineering (same set you used) ---
    titles = df["Name"].str.extract(r",\s*([^\.]+)\.")[0].str.strip().fillna("Unknown")
    common = {"Mr","Mrs","Miss","Master"}
    df["Title"] = titles.where(titles.isin(common), "Rare")
    df["FamilySize"] = df["SibSp"].fillna(0) + df["Parch"].fillna(0) + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

    target = "Survived"
    features = ["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked",
                "Title","FamilySize","IsAlone"]

    X = df[features].copy()
    y = df[target].copy()

    num_cols = ["Age","SibSp","Parch","Fare","FamilySize","IsAlone"]
    cat_cols = ["Pclass","Sex","Embarked","Title"]

    # Trees don’t need scaling; still impute and one-hot categoricals
    numeric = SimpleImputer(strategy="median")
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    prep = ColumnTransformer([("num", numeric, num_cols),
                              ("cat", categorical, cat_cols)])

    rf = RandomForestClassifier(
        n_estimators=600,
        max_depth=None,
        min_samples_split=2,
        random_state=SEED,
        n_jobs=-1,
    )

    pipe = Pipeline([("prep", prep), ("clf", rf)])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    scores = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
    print(f"RF 5-fold CV accuracy: {scores.mean():.4f} ± {scores.std():.4f}")
    print("Fold scores:", [round(float(s), 4) for s in scores])

if __name__ == "__main__":
    main()
