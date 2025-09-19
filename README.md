# Titanic ML — learning-by-doing

Predicting passenger survival on the Titanic (Kaggle). This repo shows a clean, reproducible ML workflow:
- Baseline heuristic → Logistic Regression with feature engineering → (compare) Random Forest
- Proper cross-validation with scikit-learn Pipelines (no leakage)
- Small Streamlit demo powered by the trained pipeline

## Quickstart (Docker, recommended)

```bash
# from project root
docker build -t titanic-ml:dev .
# run an interactive shell in the dev image
docker run --rm -it -v "$PWD":/app -w /app -p 8501:8501 --entrypoint bash titanic-ml:dev
```

## Inside the container:
```bash
# 1) sanity check CV
python src/logreg_cv.py

# 2) train & save model + write submission
python src/train_logreg.py

# 3) run the demo app (open http://localhost:8501 in Windows browser)
streamlit run app/streamlit_app.py
```

## Results (so far)

Baseline (female→survived): ~0.7868 train accuracy

LogReg (no FE), 5-fold CV: ~0.7969 ± 0.0146

LogReg + features (Title, FamilySize, IsAlone), 5-fold CV: ~0.8305 ± 0.0084

Random Forest (defaultish), 5-fold CV: ~0.8126 ± 0.0136

See reports/results.md for details.

## Repo structure
```bash
data/raw/                # train.csv, test.csv (not committed)
src/
  baseline.py            # heuristic baseline
  features.py            # feature engineering helpers
  logreg_cv.py           # logistic regression with CV
  logreg_tune.py         # quick C tuning
  train_logreg.py        # fit full model + save + write submission
  rf_cv.py               # random forest with CV
app/
  streamlit_app.py       # tiny demo UI
reports/
  results.md             # running notes & scores
Dockerfile
.dockerignore
requirements.txt
```

## What I learned

Why we always set a baseline first.

How to build leakage-safe Pipelines with ColumnTransformer.

Why one-hot + scaling matter for linear models.

Cross-validation (Stratified K-Fold) for reliable estimates.

Minimal feature engineering that actually lifts performance.

## Author
Manoj Nawale
manoj.nawale.work@gmail.com