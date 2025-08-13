import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Concrete Strength Predictor", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv("concrete.csv")
    return df

@st.cache_resource
def load_artifacts():
    model = joblib.load("model.joblib")
    with open("metadata.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    return model, meta

df = load_data()
model, meta = load_artifacts()

st.title("🧱 Concrete Compressive Strength — ML App")
st.caption("Deployed with Streamlit. Upload-free demo using the provided dataset.")

tab1, tab2, tab3 = st.tabs(["Overview", "Explore", "Predict"])

with tab1:
    st.subheader("Model Summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", meta["n_rows"])
    col2.metric("Features", meta["n_features"])
    col3.metric("R² (test)", f'{meta["r2"]:.3f}')
    st.metric("RMSE (test)", f'{meta["rmse"]:.2f} MPa')

    st.write("**Features used:**", ", ".join(meta["features"]))
    st.write("This model is a GradientBoostingRegressor trained on an 80/20 split.")

with tab2:
    st.subheader("Dataset Preview")
    st.dataframe(df.head(20), use_container_width=True)

    st.subheader("Correlation (Pearson)")
    corr = df.corr(numeric_only=True)
    fig, ax = plt.subplots()
    cax = ax.matshow(corr, interpolation="nearest")
    fig.colorbar(cax)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90)
    ax.set_yticklabels(corr.columns)
    st.pyplot(fig)

with tab3:
    st.subheader("Make a Prediction")
    st.write("Enter mix design parameters to estimate compressive strength (MPa).")

    inputs = {}
    for col in meta["features"]:
        min_v = float(df[col].min())
        max_v = float(df[col].max())
        default_v = float(df[col].median())
        step = (max_v - min_v) / 100 if max_v > min_v else 1.0
        inputs[col] = st.number_input(
            col, min_value=min_v, max_value=max_v, value=default_v, step=step, format="%.2f"
        )

    if st.button("Predict Strength"):
        X = np.array([[inputs[c] for c in meta["features"]]], dtype=float)
        pred = float(model.predict(X)[0])
        st.success(f"Estimated compressive strength: **{pred:.2f} MPa**")
