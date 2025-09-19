import joblib
import pandas as pd
import streamlit as st
from pathlib import Path

st.set_page_config(page_title="Titanic Survival Predictor", layout="centered")
st.title("🛳️ Titanic Survival Predictor")

MODEL_PATH = Path("titanic_model.pkl")
if not MODEL_PATH.exists():
    st.error("Model not found. Run `python src/train_logreg.py` first.")
    st.stop()

pipe = joblib.load(MODEL_PATH)

st.markdown("Enter passenger details:")

col1, col2, col3 = st.columns(3)
with col1:
    pclass = st.selectbox("Passenger Class (Pclass)", [1, 2, 3], index=2)
    age = st.number_input("Age", min_value=0.0, max_value=90.0, value=29.0, step=1.0)
with col2:
    sex = st.selectbox("Sex", ["male", "female"])
    sibsp = st.number_input("Siblings/Spouses Aboard (SibSp)", min_value=0, max_value=8, value=0, step=1)
with col3:
    embarked = st.selectbox("Embarked", ["S", "C", "Q"])
    parch = st.number_input("Parents/Children Aboard (Parch)", min_value=0, max_value=6, value=0, step=1)

fare = st.number_input("Fare", min_value=0.0, max_value=600.0, value=32.2, step=0.1)

# Match training features (we engineered these implicitly during training)
family_size = sibsp + parch + 1
is_alone = 1 if family_size == 1 else 0
# Title isn’t known at inference time; offer a simple selector
title = st.selectbox("Title", ["Mr", "Mrs", "Miss", "Master", "Rare", "Unknown"], index=0)

if st.button("Predict"):
    row = pd.DataFrame([{
        "Pclass": pclass,
        "Sex": sex,
        "Age": age,
        "SibSp": sibsp,
        "Parch": parch,
        "Fare": fare,
        "Embarked": embarked,
        "Title": title,
        "FamilySize": family_size,
        "IsAlone": is_alone,
    }])
    pred = int(pipe.predict(row)[0])
    prob = float(pipe.predict_proba(row)[0,1])
    st.success(f"Prediction: {'Survived (1)' if pred==1 else 'Not Survived (0)'}")
    st.caption(f"Model confidence (P[survive]): {prob:.3f}")
