#Smart Home Sensor Forecasting with AI and Live Data Visualization

This project demonstrates an end-to-end AI and machine learning pipeline for smart home sensor forecasting and occupancy prediction. It combines classical machine learning models, ensemble methods, and deep learning (LSTM) with an interactive Streamlit dashboard for data visualization and real-time inference.

The project is structured to reflect real-world practice: models are trained offline, saved to disk, and then reused for fast and stable inference in a live application.

What This Project Shows:
Practical use of machine learning and deep learning for time-series data
Multi-output forecasting of environmental sensors and occupancy
Comparison of classical ML models and neural networks
Clean separation between training and deployment
Interactive data visualization and live prediction
Reproducible and deployment-oriented design

Data and Prediction Task:
The dataset contains smart home sensor readings, including:
Temperature
Humidity
Light
CO₂
Humidity Ratio
Occupancy

Given historical sensor measurements, the task is to predict the next time-step values for all sensors simultaneously, with occupancy treated as a binary variable. This setup is representative of common problems in smart homes, IoT systems, and building analytics.

Project Structure
.
├── train.py
├── Smarthome_Sensor_Predictions_Live.py
├── datatraining.txt
├── datatest.txt
├── models/


train.py: trains and saves all machine learning and deep learning models
Streamlit app: loads trained models and performs visualization and live prediction
models/: stored models and metadata used during inference

Model Training (train.py)

The training script implements a multi-output forecasting pipeline using a variety of approaches:
Linear models (Linear Regression, Ridge, Lasso, ElasticNet)
Instance-based models (KNN)
Tree-based models (Decision Tree, Extra Trees)
Ensemble methods (Random Forest, Gradient Boosting)
Neural networks (MLP)
Deep learning with LSTM for temporal modeling
Classical models are trained with feature scaling and multi-output regression.
The LSTM model uses a sliding window to capture temporal dependencies in sensor data.

All trained models and metadata are saved to disk and reused by the Streamlit application.

To train the models:
python train.py

Streamlit Application: Visualization and Live Prediction
The Streamlit app focuses on understanding the data and observing model behavior.
Features
Exploratory data analysis (EDA)
Summary statistics and correlations
Time-series visualization
Distribution and box plots
Live streaming predictions using trained models
Adjustable inference speed
Comparison of true vs predicted values in real time
Occupancy predictions are handled as binary outputs, reflecting real-world use cases.

To run the app:
streamlit run Smarthome_Sensor_Predictions_Live.py

Tools and Technologies
Python
Scikit-learn
TensorFlow / Keras
Pandas, NumPy
Streamlit
Matplotlib

Why This Project Is Relevant
This project demonstrates skills that are directly applicable to AI Engineer and Data Scientist roles, including:
Designing end-to-end AI systems
Training and managing multiple models
Time-series forecasting
Deep learning with LSTMs
Data visualization and communication
Building reusable and deployment-ready ML pipelines



This repository is intended for learning, research, and portfolio demonstration.
It can be extended with evaluation metrics, model comparison tables, feature importance analysis, or cloud deployment.
