# 🏗️ PHI - Project Health Index Suite

An AI-powered tool for project management and engineering feasibility analysis. This application uses a Neural Network (MLPRegressor) to predict project cost and schedule risks, providing a Project Health Index (PHI) based on Monte Carlo simulations.

## 🚀 Features
- **AI Forecasting:** Predicts final project cost and duration using historical data and stage-specific risk profiles.
- **Dynamic PHI Dashboard:** Real-time health scoring based on Cost, Time, and Risk metrics.
- **Monte Carlo Simulations:** Utilizes P90 confidence intervals for stage-specific distribution of resources.
- **Multi-Stage Tracking:** Support for both predictive (planning) and tracking (execution) modes.

## 🛠️ Setup & Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/phi-project.git
   cd phi-project
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the model (Optional):**
   If you want to retrain the model with the latest dataset:
   ```bash
   python train_model.py
   ```

4. **Run the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

## 📊 How it Works
The PHI logic combines three key performance indicators:
- **CPI (Cost Performance Index):** Earned Value / Actual Cost
- **SPI (Schedule Performance Index):** Planned Schedule / Actual Schedule
- **Risk Assessment:** Aggregated risk scores across project stages.

The tool calculates a weighted index (%) to help project managers identify potential overruns before they occur.

## 📂 File Structure
- `app.py`: The Streamlit web application.
- `train_model.py`: Script to process data and train the neural network.
- `phi_model.pkl`: Pre-trained MLPRegressor model.
- `scaler_x.pkl` & `scaler_y.pkl`: Pre-fitted scalers for normalization.
- `Engineering_Cost_Feasibility_Dataset.csv`: The base dataset for training.

## 📝 Requirements
- Python 3.8+
- Streamlit
- Pandas
- Scikit-learn
- Joblib
- Numpy

---
Created by [Your Name]
