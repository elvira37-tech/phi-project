# Predictive Earned Value Management (EVM) Tool

This project provides an AI-powered tool for predicting construction project costs and timelines using JAX/Flax. It integrates traditional Earned Value Management (EVM) metrics with neural network-based forecasting to provide more accurate project outlooks.

## 🏗️ Features
- **AI Forecasting:** Uses a neural network trained on historical construction data to predict costs and durations for future project stages.
- **Interactive Dashboard:** Built with Streamlit, allowing users to input project parameters (type, area, budget) and track progress in real-time.
- **Predictive EVM Metrics:** Calculates CPI (Cost Performance Index), SPI (Schedule Performance Index), and EAC (Estimate at Completion) based on both actual progress and AI predictions.
- **Visual S-Curves:** Interactive Plotly charts comparing the baseline plan, actual performance, and AI-driven forecasts.

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- Pip

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/phi-project.git
   cd phi-project
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Tool
To launch the interactive dashboard:
```bash
streamlit run app.py
```

## 📂 Project Structure
- `app.py`: The main Streamlit application for the interactive dashboard.
- `pm_project.py`: The training and data generation script (originally developed in Google Colab).
- `project_model.pkl`: Pre-trained model weights and preprocessing assets.
- `*.csv`: Datasets used for training and validation.
- `validation_results.npz`: Detailed metrics from the model evaluation.

## 🧠 Model Architecture
The tool uses a multi-layer perceptron (MLP) built with **JAX** and **Flax NNX**. It processes categorical project types, complexity levels, and numerical scope parameters to forecast stage-by-stage actuals.

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
