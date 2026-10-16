import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Bankruptcy Prevention", page_icon="📊")

MODEL_PATH = "saved_models/best_model.pkl"
ENCODER_PATH = "saved_models/label_encoder.pkl"

st.title("Bankruptcy Prevention System")
st.write("Enter the company's risk indicators to predict its class.")

if not os.path.exists(MODEL_PATH) or not os.path.exists(ENCODER_PATH):
    st.error("Model files not found. Check the saved_models folder.")
    st.stop()

model = joblib.load(MODEL_PATH)
encoder = joblib.load(ENCODER_PATH)

features = [
    "industrial_risk",
    "management_risk",
    "financial_flexibility",
    "credibility",
    "competitiveness",
    "operating_risk"
]

values = {}
for feature in features:
    values[feature] = st.selectbox(
        feature.replace("_", " ").title(),
        [0.0, 0.5, 1.0]
    )

if st.button("Predict Bankruptcy Risk"):
    input_data = pd.DataFrame([values], columns=features)
    prediction = model.predict(input_data)

    try:
        result = encoder.inverse_transform(prediction.astype(int))[0]
    except (ValueError, TypeError):
        result = prediction[0]

    st.subheader("Prediction Result")
    if str(result).upper() == "BANKRUPTCY":
        st.error(f"Predicted class: {result}")
    else:
        st.success(f"Predicted class: {result}")

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_data)[0]
        st.write("Prediction probabilities")
        st.bar_chart(pd.Series(probabilities, index=model.classes_))
