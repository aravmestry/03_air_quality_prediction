import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from pathlib import Path

st.set_page_config(
    page_title="Air Quality Prediction Studio",
    page_icon="🌫️",
    layout="wide"
)

OUTPUT_DIR = Path("outputs")
MODEL_DIR = Path("models")

@st.cache_data
def load_data():
    df = pd.read_csv(OUTPUT_DIR / "air_quality_clean.csv")
    return df

@st.cache_resource
def load_model():
    return joblib.load(MODEL_DIR / "best_model.pkl")

df = load_data()
model = load_model()

features = [
    "PM10", "SO2", "NO2", "CO", "O3",
    "TEMP", "PRES", "DEWP", "RAIN", "WSPM",
    "season", "weekend", "hour_sin", "hour_cos"
]

st.title("🌫️ Air Quality Prediction Studio")
st.markdown(
    """
    This dashboard predicts **PM2.5 concentration** using pollutant and meteorological variables
    from the Beijing Multi-Site Air Quality Dataset.
    """
)

tab1, tab2, tab3, tab4 = st.tabs([
    "Overview",
    "Prediction Tool",
    "Model Results",
    "Data Explorer"
])

with tab1:
    st.subheader("Dataset Overview")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Observations", f"{len(df):,}")
    c2.metric("Stations", df["station"].nunique())
    c3.metric("Target", "PM2.5")
    c4.metric("Best Model", "Extra Trees")

    st.subheader("PM2.5 Distribution")
    fig = px.histogram(
        df,
        x="PM2.5",
        nbins=80,
        title="Distribution of PM2.5 Concentrations"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Average PM2.5 by Station")
    station_avg = df.groupby("station", as_index=False)["PM2.5"].mean()
    station_avg = station_avg.sort_values("PM2.5", ascending=False)

    fig2 = px.bar(
        station_avg,
        x="station",
        y="PM2.5",
        title="Average PM2.5 Concentration by Monitoring Station"
    )
    st.plotly_chart(fig2, use_container_width=True)

with tab2:
    st.subheader("Predict PM2.5 Concentration")

    st.write("Adjust the environmental variables below to estimate PM2.5 concentration.")

    user_inputs = {}

    col1, col2 = st.columns(2)

    for i, feature in enumerate(features):
        active_col = col1 if i % 2 == 0 else col2

        min_val = float(df[feature].min())
        max_val = float(df[feature].max())
        mean_val = float(df[feature].mean())

        if min_val == max_val:
            active_col.write(f"{feature}: {min_val}")
            user_inputs[feature] = min_val
        else:
            user_inputs[feature] = active_col.slider(
                feature,
                min_value=min_val,
                max_value=max_val,
                value=mean_val
            )

    input_df = pd.DataFrame([user_inputs])
    prediction = model.predict(input_df)[0]

    st.divider()
    st.metric("Predicted PM2.5 Concentration", f"{prediction:.2f} µg/m³")

    if prediction <= 12:
        category = "Good"
    elif prediction <= 35.4:
        category = "Moderate"
    elif prediction <= 55.4:
        category = "Unhealthy for Sensitive Groups"
    elif prediction <= 150.4:
        category = "Unhealthy"
    elif prediction <= 250.4:
        category = "Very Unhealthy"
    else:
        category = "Hazardous"

    st.subheader(f"Air Quality Category: {category}")

with tab3:
    st.subheader("Model Comparison")

    comparison_path = OUTPUT_DIR / "model_comparison.csv"

    if comparison_path.exists():
        results = pd.read_csv(comparison_path)
        st.dataframe(results, use_container_width=True)

        fig3 = px.bar(
            results.sort_values("R2", ascending=False),
            x="Model",
            y="R2",
            title="Model Comparison by R²"
        )
        st.plotly_chart(fig3, use_container_width=True)

        fig4 = px.bar(
            results.sort_values("RMSE", ascending=True),
            x="Model",
            y="RMSE",
            title="Model Comparison by RMSE"
        )
        st.plotly_chart(fig4, use_container_width=True)
    else:
        st.warning("Model comparison file not found. Run compare_models.py first.")

    st.subheader("Feature Importance")

    importance_path = OUTPUT_DIR / "feature_importance.csv"

    if importance_path.exists():
        imp = pd.read_csv(importance_path)
        fig5 = px.bar(
            imp.sort_values("Importance"),
            x="Importance",
            y="Feature",
            orientation="h",
            title="Random Forest Feature Importance"
        )
        st.plotly_chart(fig5, use_container_width=True)
    else:
        st.warning("Feature importance file not found.")

with tab4:
    st.subheader("Cleaned Air Quality Dataset")
    st.dataframe(df.head(1000), use_container_width=True)

    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Cleaned Dataset",
        data=csv,
        file_name="cleaned_air_quality_dataset.csv",
        mime="text/csv"
    )

st.divider()
st.caption("Project 3: Air Quality Prediction Studio | Climate and Ecology Modeling Portfolio")