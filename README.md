# Project Health Indicator (PHI) Assessment Suite

This repository contains an AI-driven tool for assessing the feasibility and health of engineering projects based on cost, schedule, and risk factors.

## Features
- **AI-Powered Predictions:** Uses a neural network (MLPRegressor) to forecast project duration and cost.
- **Monte Carlo Simulations:** Weights stage-specific risks and costs using statistical distributions.
- **Project Health Index (PHI):** A consolidated metric for tracking project status (Feasible, Borderline, or Not Feasible).
- **Multiple Interfaces:**
  - **CLI Tool:** `phi_cli.py` for quick analysis.
  - **Streamlit App:** `phi_tool.py` for a web interface.
  - **Jupyter Notebook:** `PM_project.ipynb` for detailed research and model training.

## Installation
```bash
pip install -r requirements.txt
```

## Usage (CLI)
```bash
python phi_cli.py --area 1200 --complexity 2 --resources 85 --days 365
```

## Web Interface
```bash
streamlit run phi_tool.py
```
