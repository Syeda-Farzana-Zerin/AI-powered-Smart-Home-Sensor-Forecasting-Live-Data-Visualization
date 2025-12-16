# -------------------------------------------------------------
#  TRAIN ALL 20 MODELS AND SAVE THEM TO DISK FOR STREAMLIT APP
# -------------------------------------------------------------

import numpy as np
import pandas as pd
import joblib
import os

from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, SGDRegressor, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
)
from sklearn.neural_network import MLPRegressor

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


# =============================================================
#  Settings
# =============================================================

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

FEATURE_COLS = ["Temperature", "Humidity", "Light", "CO2", "HumidityRatio", "Occupancy"]
WINDOW = 5


# =============================================================
#  Data Loading
# =============================================================

def load_data():
    train = pd.read_csv("datatraining.txt")

    if "date" not in train.columns:
        train.columns = ["idx", "date", "Temperature", "Humidity",
                         "Light", "CO2", "HumidityRatio", "Occupancy"]
        train = train.drop(columns=["idx"])

    train["date"] = pd.to_datetime(train["date"])
    return train


# =============================================================
#  LSTM sequence creation
# =============================================================

def make_sequences(df, window=5):
    df = df.sort_values("date").reset_index(drop=True)
    vals = df[FEATURE_COLS].values
    X_seq, y_seq = [], []

    for i in range(window, len(vals)):
        X_seq.append(vals[i-window:i])
        y_seq.append(vals[i])

    return np.array(X_seq), np.array(y_seq)


# =============================================================
#  LSTM Multi-output class
# =============================================================

class LSTM_MultiOutput:
    def __init__(self, window=5, units=64, epochs=10):
        self.window = window
        self.units = units
        self.epochs = epochs
        self.model = None

    def fit(self, X, y):
        model = Sequential()
        model.add(LSTM(self.units, input_shape=(self.window, 6)))
        model.add(Dense(6))
        model.compile(optimizer="adam", loss="mse")
        model.fit(X, y, epochs=self.epochs, batch_size=32, verbose=0)
        self.model = model

    def save(self, path):
        self.model.save(path)

    def predict(self, X):
        return self.model.predict(X)


# =============================================================
#  TRAINING
# =============================================================

def train_and_save():
    train_df = load_data()

    # Classical model training data
    X_train = train_df[FEATURE_COLS].values[:-1]
    y_train = train_df[FEATURE_COLS].values[1:]

    # LSTM data
    X_train_seq, y_train_seq = make_sequences(train_df, WINDOW)

    # Multi-output wrapper
    def pipe(reg):
        return Pipeline([
            ("scaler", StandardScaler()),
            ("multi", MultiOutputRegressor(reg))
        ])

    # 19 classical models + 1 LSTM
    models = {
        # Linear (4)
        "LinearRegression": pipe(LinearRegression()),
        "Ridge": pipe(Ridge()),
        "Lasso": pipe(SGDRegressor(penalty="l1", max_iter=2000)),
        "ElasticNet": pipe(SGDRegressor(penalty="elasticnet", max_iter=2000)),

        # KNN (3)
        "KNN3": pipe(KNeighborsRegressor(3)),
        "KNN5": pipe(KNeighborsRegressor(5)),
        "KNN10": pipe(KNeighborsRegressor(10)),

        # Tree-based (2)
        "DecisionTree": pipe(DecisionTreeRegressor()),
        "ExtraTree": pipe(ExtraTreesRegressor(n_estimators=80)),

        # Ensemble (3)
        "RF100": pipe(RandomForestRegressor(n_estimators=100)),
        "RF300": pipe(RandomForestRegressor(n_estimators=300)),
        "ExtraTrees200": pipe(ExtraTreesRegressor(n_estimators=200)),

        # Boosting (3)
        "GradientBoosting": pipe(GradientBoostingRegressor()),
        "AdaBoostStyle": pipe(GradientBoostingRegressor()),
        "BaggingStyle": pipe(RandomForestRegressor(n_estimators=50)),

        # Robust Linear (3)
        "SGD": pipe(SGDRegressor(max_iter=2000)),
        "Huber": pipe(SGDRegressor(loss="huber", max_iter=2000)),
        "PassiveAggressive": pipe(SGDRegressor(loss="epsilon_insensitive", max_iter=2000)),

        # MLP
        "MLP": pipe(MLPRegressor(hidden_layer_sizes=(64,32), max_iter=500))
    }

    # Train + Save classical models
    print("Training classical models...")
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        joblib.dump(model, f"{MODEL_DIR}/{name}.pkl")

    # Train + save LSTM
    print("Training LSTM...")
    lstm = LSTM_MultiOutput(window=WINDOW)
    lstm.fit(X_train_seq, y_train_seq)
    lstm.save(f"{MODEL_DIR}/LSTM.h5")

    # Save metadata
    joblib.dump(FEATURE_COLS, f"{MODEL_DIR}/feature_cols.pkl")
    joblib.dump(WINDOW, f"{MODEL_DIR}/window.pkl")

    print("All models saved successfully!")


# Run training
if __name__ == "__main__":
    train_and_save()
