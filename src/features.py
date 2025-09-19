import pandas as pd
import re
import numpy as np

def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    titles = out["Name"].str.extract(r",\s*([^\.]+)\.")[0].str.strip().fillna("Unknown")
    common = {"Mr","Mrs","Miss","Master"}
    out["Title"] = titles.where(titles.isin(common), "Rare")

    out["FamilySize"] = out["SibSp"].fillna(0) + out["Parch"].fillna(0) + 1
    out["IsAlone"] = (out["FamilySize"] == 1).astype(int)

    # NEW features
    out["HasCabin"] = out["Cabin"].notna().astype(int)
    out["FareLog"]  = np.log1p(out["Fare"].fillna(0))

    return out
