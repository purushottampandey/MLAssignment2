import streamlit as st
import pandas as pd
import joblib
import numpy as np
from pathlib import Path

# -----------------------------------
# Page Config
# -----------------------------------
st.set_page_config(page_title="Thyroid Prediction App", layout="centered")

# Base paths (adjust if you move files)
BASE = Path("/Users/purushottampandey/Documents/V S Code/MLASSIGNMENT")
MODEL_PATH = BASE / "thyroid_rf_model.pkl"
SCALER_PATH = BASE / "scaler.pkl"
ENCODER_PATH = BASE / "label_encoder.pkl"

# -----------------------------------
# Load Model + Scaler + Encoders (cached)
# -----------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    enc = None
    try:
        enc = joblib.load(ENCODER_PATH)
    except Exception:
        enc = None
    return model, scaler, enc

model, scaler, encoders = load_artifacts()

# -----------------------------------
# Encoding Helper (robust for multiple saved formats)
# -----------------------------------
def encode_value(column, value):
    """
    Handles:
    - encoders as a dict mapping column -> LabelEncoder
    - encoders as a single LabelEncoder (fallback)
    - missing encoders: use deterministic local mapping consistent with training
    """
    # Local deterministic mappings (match notebook preprocessing)
    binary_map = {"f": 0, "t": 1}
    sex_map = {"F": 0, "M": 1}
    referral_map = {"SVHC": 0, "SVI": 1, "STMW": 2, "other": 3}

    try:
        # If we have a dict of encoders (preferred)
        if isinstance(encoders, dict) and column in encoders:
            le = encoders[column]
            return int(le.transform([str(value)])[0])

        # If it's a LabelEncoder instance (single), try to use it
        from sklearn.preprocessing import LabelEncoder

        if isinstance(encoders, LabelEncoder):
            # best-effort: transform; if fails, fall back to local maps
            try:
                return int(encoders.transform([str(value)])[0])
            except Exception:
                pass

        # Fallback deterministic mappings
        if column == "sex":
            return sex_map.get(value, value)
        if column in {"on_thyroxine", "pregnant", "sick", "goitre", "tumor"}:
            return binary_map.get(value, value)
        if column == "referral_source":
            return referral_map.get(value, value)

        # If nothing matched, return value as-is (numeric columns)
        return value
    except Exception as e:
        st.error(f"Encoding error for column '{column}': {e}")
        return np.nan

# -----------------------------------
# UI
# -----------------------------------
st.title("Thyroid Disease Prediction")

st.subheader("👤 Patient Information")
age = st.number_input("Age", 1, 120, 30)
sex = st.selectbox("Sex", ["F", "M"])
pregnant = st.selectbox("Pregnant", ["t", "f"])
sick = st.selectbox("Sick", ["t", "f"])
on_thyroxine = st.selectbox("On Thyroxine", ["t", "f"])

st.subheader("🩺 Medical Conditions")
goitre = st.selectbox("Goitre", ["t", "f"])
tumor = st.selectbox("Tumor", ["t", "f"])

st.subheader("🧪 Lab Test Results")
TSH = st.number_input("TSH Level", 0.0, 100.0, 1.5)
T3 = st.number_input("T3 Level", 0.0, 10.0, 2.0)
TT4 = st.number_input("TT4 Level", 0.0, 300.0, 100.0)
T4U = st.number_input("T4U Level", 0.0, 5.0, 1.0)
FTI = st.number_input("FTI Level", 0.0, 500.0, 100.0)

st.subheader("📋 Other Information")
referral_source = st.selectbox("Referral Source", ["SVI", "SVHC", "STMW", "other"])

# -----------------------------------
# Prepare Input (match training feature order)
# -----------------------------------
feature_order = [
    "age", "sex", "on_thyroxine", "pregnant", "sick",
    "goitre", "tumor", "TSH", "T3", "TT4", "T4U", "FTI", "referral_source"
]

raw = {
    "age": age,
    "sex": encode_value("sex", sex),
    "on_thyroxine": encode_value("on_thyroxine", on_thyroxine),
    "pregnant": encode_value("pregnant", pregnant),
    "sick": encode_value("sick", sick),
    "goitre": encode_value("goitre", goitre),
    "tumor": encode_value("tumor", tumor),
    "TSH": TSH,
    "T3": T3,
    "TT4": TT4,
    "T4U": T4U,
    "FTI": FTI,
    "referral_source": encode_value("referral_source", referral_source)
}

input_data = pd.DataFrame([raw], columns=feature_order)

# -----------------------------------
# Prediction
# -----------------------------------
if st.button("🔍 Predict Thyroid Class"):
    if input_data.isnull().values.any():
        st.error("❌ Invalid or missing value detected after encoding.")
    else:
        try:
            # scaler expects numeric DataFrame in the same order used during training
            scaled_input = scaler.transform(input_data)
            prediction = model.predict(scaled_input)[0]

            st.subheader("📊 Prediction Result")
            if int(prediction) == 1:
                st.error("⚠️ Thyroid Disease Detected")
            else:
                st.success("✅ Normal")

            # Confidence Score
            #if hasattr(model, "predict_proba"):
               # probs = model.predict_proba(scaled_input)[0]
                #confidence = float(np.max(probs))
                #st.write(f"**Confidence:** {confidence:.2%}")
                #st.progress(float(confidence))

        except Exception as e:
            st.error(f"❌ Prediction Error: {e}")

# -----------------------------------
# Footer
# -----------------------------------
st.markdown("---")
st.caption("ML Classification Assignment • Streamlit App")
