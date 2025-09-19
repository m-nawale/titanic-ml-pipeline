import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

TRAIN = "data/raw/train.csv"
OUTDIR = Path("reports/figures")
OUTDIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(TRAIN)

# 1) Missingness bar
na = df.isna().mean().sort_values(ascending=False)
plt.figure()
na.plot.bar(rot=45)
plt.ylabel("Fraction missing")
plt.tight_layout()
plt.savefig(OUTDIR / "00_missingness.png")
plt.close()

# 2) Survival by Sex
plt.figure()
(df.groupby("Sex")["Survived"].mean().sort_values()).plot.bar(rot=0)
plt.ylabel("Survival rate")
plt.title("Survival by Sex")
plt.tight_layout()
plt.savefig(OUTDIR / "01_survival_by_sex.png")
plt.close()

# 3) Survival by Pclass
plt.figure()
(df.groupby("Pclass")["Survived"].mean()).plot.bar(rot=0)
plt.ylabel("Survival rate")
plt.title("Survival by Pclass")
plt.tight_layout()
plt.savefig(OUTDIR / "02_survival_by_pclass.png")
plt.close()

# 4) Age distribution split by Survived (overlay hist)
plt.figure()
df[df["Survived"]==0]["Age"].dropna().plot.hist(bins=20, alpha=0.6, label="Not Survived")
df[df["Survived"]==1]["Age"].dropna().plot.hist(bins=20, alpha=0.6, label="Survived")
plt.xlabel("Age"); plt.ylabel("Count"); plt.legend()
plt.title("Age distribution by outcome")
plt.tight_layout()
plt.savefig(OUTDIR / "03_age_hist_by_outcome.png")
plt.close()

print("Saved figures to", OUTDIR)
