import joblib
import numpy as np

MODEL = "titanic_model.pkl"

def main():
    pipe = joblib.load(MODEL)
    prep = pipe.named_steps["prep"]
    clf  = pipe.named_steps["clf"]

    # Get feature names after ColumnTransformer + OneHotEncoder
    num_feats = prep.transformers_[0][2]  # numeric columns list
    cat_pipe  = prep.transformers_[1][1]  # Pipeline(imputer, onehot)
    cat_cols  = prep.transformers_[1][2]  # categorical columns list
    ohe = cat_pipe.named_steps["onehot"]
    ohe_feats = ohe.get_feature_names_out(cat_cols)

    feature_names = list(num_feats) + list(ohe_feats)

    coefs = clf.coef_.ravel()
    pairs = sorted(zip(feature_names, coefs), key=lambda x: abs(x[1]), reverse=True)

    print("Top + and - coefficients (magnitude):")
    for name, val in pairs[:20]:
        print(f"{name:25s} {val:+.3f}")
    print("\nNote: + pushes toward Survived=1, - toward 0. Magnitude shows strength (after scaling/one-hot).")

if __name__ == "__main__":
    main()
