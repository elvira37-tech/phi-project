# PHI - Project Intelligence Dashboard

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://phi-project-bvpv4u68spqggbfdadys35.streamlit.app/)

**PHI (Project Health Intelligence)** is an advanced analytics platform designed for engineering project management. It leverages Neural Networks (built with JAX and Flax) to predict stage-by-stage costs and durations, providing project managers with data-driven insights to mitigate risks and optimize resource allocation.

## 🚀 Key Features

- **AI-Driven Estimation**: Predict baseline costs and durations for 5 critical construction stages:
  - Site Preparation
  - Foundations
  - Structural Works
  - Systems Installation
  - Finishing Works
- **Hybrid Tracking System**: Seamlessly combine manual user estimates with AI-driven stage distributions for enhanced accuracy.
- **Dynamic EVM Dashboard**: Real-time monitoring of Cost Performance Index (CPI), Schedule Performance Index (SPI), and Estimate at Completion (EAC).
- **Multi-Sector Support**: Pre-configured for Building, Bridge, Power Plant, Water Infrastructure, and Road projects.
- **Risk Assessment**: Integrated complexity and risk scoring to adjust project forecasts.

## 🛠️ Tech Stack

- **Machine Learning**: JAX, Flax, Optax (Neural Network architecture)
- **Data Processing**: Pandas, NumPy
- **Dashboard**: Streamlit
- **Serialization**: Joblib
- **Visualization**: Matplotlib

## 📦 Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/elvira37-tech/phi-project.git
   cd phi-project
   ```

2. **Set up a Virtual Environment** (Recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 🏃 Getting Started

### 1. Training the Model
Before running the dashboard, you must generate the model assets by training the neural network on the provided dataset.

```bash
python train_model.py
```
This will process `Engineering_Cost_Feasibility_Dataset.csv` and generate `jax_project_model.pkl`.

### 2. Launching the Dashboard
Start the Streamlit application:

```bash
streamlit run app.py
```

## 📂 Project Structure

- `app.py`: Main Streamlit application and UI logic.
- `train_model.py`: Data synthesis, preprocessing, and JAX/Flax model training.
- `Engineering_Cost_Feasibility_Dataset.csv`: Raw dataset containing historical project metrics.
- `PM_project.ipynb`: Research and development notebook for the PHI algorithm.
- `requirements.txt`: List of required Python packages.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

---
*Built for the future of Engineering Project Management.*
