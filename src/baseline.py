import pandas as pd

TRAIN = "data/raw/train.csv"

def main():
    df = pd.read_csv(TRAIN)
    # baseline: predict Survived=1 for females, else 0
    pred = (df["Sex"] == "female").astype(int)
    acc = (pred == df["Survived"]).mean()
    print(f"Baseline accuracy (female=1 else 0): {acc:.4f}")
    # sanity: show class balance
    print("Survival rate in train:", df["Survived"].mean().round(4))
    # quick crosstab
    print(pd.crosstab(df["Sex"], df["Survived"], normalize="index").round(3))

if __name__ == "__main__":
    main()
