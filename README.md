# Smart Home Sensor Forecasting with AI and Live Data Visualization

This project demonstrates an end-to-end **AI and machine learning pipeline** for smart home sensor forecasting and occupancy prediction. It combines **classical machine learning models, ensemble methods, and deep learning (LSTM)** with an interactive Streamlit dashboard for **data visualization and real-time inference**.

The project is structured to reflect real-world practice: models are trained offline, saved to disk, and then reused for fast and stable inference in a live application.

---

## 🚀 What This Project Shows

- Practical use of **machine learning and deep learning** for time-series data  
- Multi-output forecasting of environmental sensors and occupancy  
- Comparison of classical ML models and neural networks  
- Clean separation between **training** and **deployment**  
- Interactive **data visualization and live prediction**  
- Reproducible and deployment-oriented design  

---
<img width="613" height="715" alt="image" src="https://github.com/user-attachments/assets/b6a98a22-5992-4232-9543-4cc5c0104340" />

## 📊 Data and Prediction Task

The dataset contains smart home sensor readings, including:
- Temperature  
- Humidity  
- Light  
- CO₂  
- Humidity Ratio  
- Occupancy  

Given historical sensor measurements, the task is to predict the **next time-step values for all sensors simultaneously**, with occupancy treated as a binary variable.

---

## 📁 Project Structure

.
├── train.py
├── Smarthome_Sensor_Predictions_Live.py
├── datatraining.txt
├── datatest.txt
├── models/

## 🧠 Model Training (`train.py`)

The training script implements a **multi-output forecasting pipeline** using:

- Linear models (Linear Regression, Ridge, Lasso, ElasticNet)  
- Instance-based models (KNN)  
- Tree-based models (Decision Tree, Extra Trees)  
- Ensemble methods (Random Forest, Gradient Boosting)  
- Neural networks (MLP)  
- Deep learning with **LSTM** for temporal modeling  

Classical models use feature scaling and multi-output regression.  
The LSTM model uses a sliding window to capture temporal dependencies.

### Train the models
python train.py

🎥 Streamlit Application:
The Streamlit app focuses on data understanding and live model behavior.

Features:
- Exploratory Data Analysis (EDA)
- Summary statistics and correlation analysis
- Time-series visualization
- Distribution and box plots
- Live streaming predictions
- Adjustable inference speed
- Real-time comparison of true vs predicted values

Run the app:

streamlit run Smarthome_Sensor_Predictions_Live.py

🛠️ Tools and Technologies:
- Python
-Scikit-learn
-TensorFlow / Keras
-Pandas, NumPy
-Streamlit
-Matplotlib

🎯 Why This Project Is Relevant:
This project demonstrates skills directly applicable to AI Engineer and Data Scientist roles, including:
- End-to-end AI system design
- Time-series forecasting
- Classical ML and deep learning
- Model lifecycle management
- Data visualization and communication
