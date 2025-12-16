# -------------------------------------------------------------
#  STREAMLIT APP — LOAD MODELS ONLY (NO TRAINING)
#  WITH COMPACT FIGURES + DATE + SMALL FONTS + BINARY OCCUPANCY
# -------------------------------------------------------------

import time
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import os

import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt


# =============================================================
#  CONSTANTS
# =============================================================

MODEL_DIR = "models"

# Load metadata
FEATURE_COLS = joblib.load(f"{MODEL_DIR}/feature_cols.pkl")
WINDOW = joblib.load(f"{MODEL_DIR}/window.pkl")

# Load classical models
CLASSICAL_MODELS = {}
for file in os.listdir(MODEL_DIR):
    if file.endswith(".pkl"):
        name = file.replace(".pkl", "")
        CLASSICAL_MODELS[name] = joblib.load(f"{MODEL_DIR}/{file}")

# Load LSTM
LSTM_MODEL = load_model(f"{MODEL_DIR}/LSTM.h5")


# =============================================================
#  DATA LOADING
# =============================================================

@st.cache_data
def load_data():
    train = pd.read_csv("datatraining.txt")
    test = pd.read_csv("datatest.txt")

    if "date" not in train.columns:
        cols = ["idx","date","Temperature","Humidity","Light","CO2","HumidityRatio","Occupancy"]
        train.columns = cols
        train = train.drop(columns=["idx"])

    if "date" not in test.columns:
        cols = ["idx","date","Temperature","Humidity","Light","CO2","HumidityRatio","Occupancy"]
        test.columns = cols
        test = test.drop(columns=["idx"])

    train["date"] = pd.to_datetime(train["date"])
    test["date"] = pd.to_datetime(test["date"])

    return train, test


# =============================================================
#  LSTM Predict Wrapper
# =============================================================

def lstm_predict(seq):
    seq = seq.reshape(1, WINDOW, 6)
    pred = LSTM_MODEL.predict(seq, verbose=0)
    return pred[0]


# =============================================================
#  STREAMLIT UI
# =============================================================

st.set_page_config(layout="wide")
st.title("📡 Multi-Output Sensor Forecasting ")

train_df, test_df = load_data()

page = st.sidebar.radio("Navigate", ["EDA", "Live Streaming"])


# =============================================================
#  EDA PAGE
# =============================================================

if page == "EDA":
    st.header("🔍 EDA")

    st.subheader("Head")
    st.dataframe(train_df.head())

    st.subheader("Tail")
    st.dataframe(train_df.tail())

    st.subheader("Stats")
    st.dataframe(train_df.describe().T)

    st.subheader("Correlation Heatmap")

    fig, ax = plt.subplots(figsize=(4, 3))
    heatmap = ax.imshow(train_df[FEATURE_COLS].corr(), cmap="coolwarm")
    ax.set_xticks(range(len(FEATURE_COLS)))
    ax.set_yticks(range(len(FEATURE_COLS)))
    ax.set_xticklabels(FEATURE_COLS, fontsize=6, rotation=45)
    ax.set_yticklabels(FEATURE_COLS, fontsize=6)
    fig.colorbar(heatmap, shrink=0.7)

    col_fig = st.columns([0.35, 0.65])[0]
    with col_fig:
        st.pyplot(fig, dpi=150, use_container_width=False)

    st.subheader("Time Series")
    st.line_chart(train_df.set_index("date")[FEATURE_COLS])

    st.subheader("Boxplots")
    for col in FEATURE_COLS:
        fig, ax = plt.subplots(figsize=(3, 2))
        ax.boxplot(train_df[col])
        ax.set_title(col, fontsize=8)
        col_fig = st.columns([0.35, 0.65])[0]
        with col_fig:
            st.pyplot(fig, dpi=150, use_container_width=False)

    st.subheader("Histograms")
    st.bar_chart(train_df[FEATURE_COLS])

    st.subheader("Pairwise Sample")
    st.dataframe(train_df[FEATURE_COLS].sample(200))


# =============================================================
#  LIVE STREAMING PAGE — COMPACT PLOTS + BINARY OCCUPANCY
# =============================================================

if page == "Live Streaming":

    st.header("🎥 Live Streaming Prediction")

    model_name = st.selectbox("Choose model", ["DecisionTree"] + list(CLASSICAL_MODELS.keys()))
    stream_speed = st.slider("Delay (sec)", 0.05, 1.0, 0.20)

    if "run" not in st.session_state:
        st.session_state.run = False

    cols = st.columns(2)
    if cols[0].button("▶ Start"):
        st.session_state.run = True
        st.session_state.idx = []
        st.session_state.true_vals = {c: [] for c in FEATURE_COLS}
        st.session_state.pred_vals = {c: [] for c in FEATURE_COLS}

    if cols[1].button("⏹ Stop"):
        st.session_state.run = False

    info_box = st.empty()
    table_box = st.empty()
    plot_box = st.empty()

    df_sorted = test_df.sort_values("date").reset_index(drop=True)

    if st.session_state.run:

        for i in range(len(df_sorted)):

            if not st.session_state.run:
                break

            row = df_sorted.iloc[i]

            # =======================
            #   PREDICTION
            # =======================
            if model_name == "LSTM":
                if i < WINDOW:
                    continue
                seq = df_sorted.iloc[i-WINDOW:i][FEATURE_COLS].values
                pred = lstm_predict(seq)
            else:
                mdl = CLASSICAL_MODELS[model_name]
                pred = mdl.predict(row[FEATURE_COLS].values.reshape(1, -1))[0]

            # ======================================================
            #  UPDATED: MAKE OCCUPANCY BINARY (0/1)
            # ======================================================
            pred[5] = 1 if pred[5] > 0.5 else 0

            # Store values
            st.session_state.idx.append(i)
            for ci, col in enumerate(FEATURE_COLS):
                st.session_state.true_vals[col].append(float(row[col]))
                st.session_state.pred_vals[col].append(float(pred[ci]))

            # =======================
            #   INFO
            # =======================
            info_box.markdown(
                f"""
                ### 🔹 Prediction Row {i}  
                **Date:** `{row['date']}`  
                """
            )

            # =======================
            #   FULL TABLE
            # =======================
            full_row = row.copy()
            for ci, col in enumerate(FEATURE_COLS):
                full_row[f"Pred_{col}"] = pred[ci]

            table_box.dataframe(full_row.to_frame())

            # ======================================================
            #  COMPACT PLOTS WITH SMALL FONTS
            # ======================================================

            plt.rcParams.update({
                "axes.titlesize": 6,
                "axes.labelsize": 6,
                "xtick.labelsize": 6,
                "ytick.labelsize": 6,
                "legend.fontsize": 5
            })

            fig, axs = plt.subplots(6, 1, figsize=(4, 5), sharex=True)
            fig.tight_layout(pad=0.7)

            for p, col in enumerate(FEATURE_COLS):
                axs[p].plot(
                    st.session_state.idx,
                    st.session_state.true_vals[col],
                    label="True",
                    linewidth=1
                )
                axs[p].plot(
                    st.session_state.idx,
                    st.session_state.pred_vals[col],
                    label="Pred",
                    linestyle="--",
                    linewidth=1
                )
                axs[p].set_title(col)
                axs[p].legend()

            col_fig = plot_box.columns([0.35, 0.65])[0]
            with col_fig:
                st.pyplot(fig, dpi=150, use_container_width=False)

            time.sleep(stream_speed)

        st.success("✔ Streaming complete")
