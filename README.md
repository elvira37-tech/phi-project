# PHI - Project Intelligence Dashboard

This project is a Project Management tool that uses AI to predict construction project outcomes and track live performance using Earned Value Management (EVM) metrics.

## Features
- **AI Prediction**: Predict stage-by-stage costs and durations for different project types (Building, Power Plant, Road, etc.) using a JAX/Flax Neural Network.
- **Hybrid EVM**: Combine user-estimated baselines with AI-driven weight distribution for more accurate performance tracking.
- **Live Monitoring**: Track Cost Performance Index (CPI), Schedule Performance Index (SPI), and Estimated at Completion (EAC) in real-time.

## Project Structure
- `app.py`: Streamlit web application.
- `train.py`: Script to preprocess data and train the AI model.
- `src/`:
  - `data_processing.py`: Logic for data cleaning and synthetic data generation.
  - `model.py`: JAX/Flax Neural Network architecture.
- `data/`: Contains the project datasets.

## Getting Started

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the Model
The Streamlit app requires a trained model file (`jax_project_model.pkl`). Run the training script to generate it:
```bash
python train.py
```

### 3. Run the App
```bash
streamlit run app.py
```

## How it Works
1. **Data Generation**: The system processes historical project data and generates synthetic samples using interpolation to increase the training set size to 21,000 samples.
2. **Neural Network**: A multi-layer perceptron (MLP) regressor implemented in JAX/Flax is trained to predict 10 outputs (5 stage costs and 5 stage durations).
3. **Performance Tracking**: During the 'Tracking' mode, the app calculates EVM metrics by comparing actual project progress against the AI-distributed baseline.

## GitHub Actions
This repository includes a CI workflow (`.github/workflows/ci.yml`) that automatically:
- Installs dependencies.
- Lints the code using `ruff`.
- Performs syntax checks on the main scripts.
