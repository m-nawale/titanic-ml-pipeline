# Titanic ML — Results Log

_A learner-friendly record of baselines, models, CV scores, and decisions._

**Last updated:** 2025-09-19

---

## Baseline (no ML)

- **Heuristic:** `Survived = 1` if `Sex == female` else `0`
- **Train accuracy:** **0.7868**
- **Class balance (train):** 0.3838 survived
- **Why keep it:** simple, fast benchmark to beat and sanity-check the dataset.

---

## Logistic Regression

**Preprocess (Pipeline):** median-impute numerics, most_frequent for categoricals, One-Hot encode categoricals, Standardize numerics.

- **5-fold CV (no FE):** **0.7969 ± 0.0146**
- **+ Feature Engineering (Title, FamilySize, IsAlone):** **0.8305 ± 0.0084**
- **C tuning (0.01 → 100):** best **C = 1.0** (peak at 0.8305)
- **Submission:** `submission_logreg_fe.csv` (418 rows)

**Interpretability (coefficients, signs make sense):**
- Positive toward survival: `Title_Master`, `Sex_female`, `Pclass_1`, `Fare(+)`
- Negative: `Title_Mr`, `Sex_male`, `Pclass_3`, `Age(+)`, `SibSp(+)`, `IsAlone`

---

## Random Forest (reference tree model)

**Preprocess:** impute + One-Hot (no scaling).  
**5-fold CV:** **0.8126 ± 0.0136**  
_Observation:_ Worse than LogReg+FE here; linear model + right features outperforms default RF.

---

## Gradient Boosting (current best)

**Features used:** `Pclass, Sex, Age, SibSp, Parch, Embarked, Title, FamilySize, IsAlone, HasCabin, FareLog`  
**Preprocess:** median impute numerics; most_frequent + One-Hot for categoricals.  
**Model:** `GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, max_depth=3, random_state=42)`

- **5-fold CV:** **0.8440 ± 0.0170**
- **Submission:** `submission_gb.csv` (418 rows)
- **Predicted survival rate (test):** 0.3541

_Notes:_ GB captures non-linearities/interactions (e.g., Sex×Pclass, age effects) without manual feature crosses; hence the lift vs LogReg.

---

## EDA findings (quick)

- **Missingness:** `Age` (~20%), `Cabin` (mostly missing), `Embarked` (few).
  - **Action:** median-impute `Age`/`Fare`, most_frequent for `Embarked`. Skip raw `Cabin` for now; use `HasCabin` flag.
- **Strong signals:**  
  - `Sex` → large survival gap (female ≫ male).  
  - `Pclass` → 1 > 2 > 3.  
  - `Age` → children higher survival; adults vary.
- **Feature hypotheses (used/considered):**  
  - **Used:** `Title` (from `Name`), `FamilySize`, `IsAlone`, `FareLog`, `HasCabin`.  
  - **Later:** age bins, fare quantiles, deck extraction (from `Cabin`), ticket group size.

**Figures:**
- `reports/figures/00_missingness.png`
- `reports/figures/01_survival_by_sex.png`
- `reports/figures/02_survival_by_pclass.png`
- `reports/figures/03_age_hist_by_outcome.png`

---

## Repro & Config

- **Seed:** 42  
- **CV:** `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`  
- **Leakage control:** all preprocessing inside `Pipeline`/`ColumnTransformer`.  
- **Commands:**
  - CV (LogReg with FE): `python src/logreg_cv.py`
  - CV (Random Forest): `python src/rf_cv.py`
  - CV (Gradient Boosting): `python src/gb_cv.py`
  - Train & save LogReg + submit: `python src/train_logreg.py`
  - Train GB + submit: `python src/train_gb.py`
  - Inspect coefficients (LogReg): `python src/inspect_logreg.py`

---

## Model Card (brief)

- **Intended use:** Educational demo for Kaggle Titanic (binary classification).  
- **Limitations:** Small historic dataset; features reflect social/contextual factors; not suitable for real-world decision making.  
- **Fairness:** Model learns associations in historic data (e.g., sex/class) → handle responsibly; predictions are illustrative.



