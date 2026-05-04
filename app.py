import streamlit as st
import pandas as pd
import joblib
import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx
import os

from src.model import ConstructionNN
from train import train_model

# --- 1. Load Assets ---
def load_jax_assets():
    file_path = 'jax_project_model.pkl'
    if not os.path.exists(file_path):
        st.warning("\u26a0\ufe0f Model file not found. This is normal for a first-time deployment.")
        if st.button("\ud83c\udfcb\ufe0f TRAIN AI MODEL NOW"):
            with st.spinner("Training model on 21k samples... This takes about 30-60 seconds."):
                try:
                    train_model()
                    st.success("\u2705 Training Complete! Refreshing...")
                    st.rerun()
                except Exception as e:
                    st.error(f"Training failed: {e}")
        st.info("\u2139\ufe0f Please click the button above to initialize the AI engine.")
        st.stop()
    
    assets = joblib.load(file_path)
    # Instantiate the model with dimensions from assets
    model = ConstructionNN(len(assets['param_cols']), len(assets['output_cols']), rngs=nnx.Rngs(0))
    nnx.state(model, nnx.Param).update(assets['model_state'])
    return model, assets

# --- 2. UI Setup ---
st.set_page_config(page_title="PHI - Project Intelligence", layout="wide")
st.title("\ud83c\udfd7\ufe0f PHI - Project Intelligence Dashboard")

try:
    model, assets = load_jax_assets()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

STAGES = ['Site_Prep', 'Foundations', 'Structure', 'Systems', 'Finishing']

st.sidebar.header("1. Mandatory Inputs")
p_type = st.sidebar.selectbox("Project Type", ['Building', 'Power Plant', 'Road', 'Bridge', 'Water Infra'])
area = st.sidebar.number_input("Engineered Area (sqm)", value=5000.0)
est_cost_input = st.sidebar.number_input("Estimated Total Cost (USD)", value=1000000.0)
est_days_input = st.sidebar.number_input("Estimated Total Time (Days)", value=300.0)

st.sidebar.header("2. Complexity & Risk")
complexity = st.sidebar.select_slider("Complexity", options=[('Low', 1), ('Medium', 2), ('High', 3)], value=2)
risk_score = st.sidebar.slider("Initial Risk Score", 0, 100, 50)
resource_score = st.sidebar.slider("Resource Score", 0, 100, 75)

mode = st.sidebar.radio("Operation Mode", ["Estimate Project", "Tracking Project"])

# --- Tracking UI ---
if mode == "Tracking Project":
    st.subheader("\ud83d\udd04 Real-Time Stage Tracking")
    c1, c2, c3 = st.columns(3)
    with c1: curr_stage = st.selectbox("Current Stage", STAGES)
    with c2: act_cost = st.number_input("Actual Cost to Date (USD)", value=0.0)
    with c3: act_days = st.number_input("Actual Days Spent in Stage", value=0.0)

if st.button("\ud83d\ude80 RUN ANALYSIS", use_container_width=True):
    # Prepare Input
    encoded_project_cols = [col for col in assets['param_cols'] if col.startswith('Proj_')]
    project_type_one_hot = np.zeros(len(encoded_project_cols), dtype=np.float32)
    
    # Project type names in assets have Proj_ prefix and underscores for spaces
    selected_proj_col = f"Proj_{p_type.replace(' ', '_')}"
    if selected_proj_col in encoded_project_cols:
        one_hot_index = encoded_project_cols.index(selected_proj_col)
        project_type_one_hot[one_hot_index] = 1.0
    else:
        # Fallback to nearest match if names differ slightly
        for i, col in enumerate(encoded_project_cols):
            if p_type.lower() in col.lower():
                project_type_one_hot[i] = 1.0
                break

    continuous_inputs = np.array([
        area,
        float(complexity),
        float(risk_score),
        float(resource_score),
        float(est_days_input)
    ], dtype=np.float32)

    raw_in = np.concatenate([project_type_one_hot, continuous_inputs]).reshape(1, -1)
    
    # Normalization
    X_scaled = (raw_in - assets['params_min']) / (assets['params_range'] + 1e-6)
    
    # Inference
    y_pred_norm = model(jnp.array(X_scaled, dtype=jnp.float32))
    y_pred = np.array(y_pred_norm) * assets['output_range'] + assets['output_min']
    
    pred_costs = np.maximum(y_pred[0, 0:5], 0.0)
    pred_times = np.maximum(y_pred[0, 5:10], 0.0)
    
    ai_total_cost = sum(pred_costs)
    ai_total_days = sum(pred_times)

    # Distribute user estimated values based on AI proportions
    stages_data = {}
    for i, s in enumerate(STAGES):
        cost_weight = pred_costs[i] / (ai_total_cost + 1e-6)
        days_weight = pred_times[i] / (ai_total_days + 1e-6)
        stages_data[s] = {
            'Cost': est_cost_input * cost_weight,
            'Days': est_days_input * days_weight
        }

    if mode == "Estimate Project":
        m1, m2 = st.columns(2)
        m1.metric("ESTIMATED BUDGET (PV)", f"${est_cost_input:,.2f}", 
                  delta=f"${(est_cost_input - ai_total_cost):,.0f} vs AI Prediction")
        m2.metric("ESTIMATED DURATION", f"{int(est_days_input)} Days", 
                  delta=f"{int(est_days_input - ai_total_days)} Days vs AI Prediction")
    else:
        # EVM Logic
        idx = STAGES.index(curr_stage)
        pv_to_date = sum(stages_data[s]['Cost'] for s in STAGES[:idx+1])
        prev_ev = sum(stages_data[s]['Cost'] for s in STAGES[:idx])
        prog = min(1.0, act_days / (stages_data[curr_stage]['Days'] + 1e-6))
        ev = prev_ev + (stages_data[curr_stage]['Cost'] * prog)
        
        cpi = ev / (act_cost + 1e-6) if act_cost > 0 else 1.0
        spi = ev / (pv_to_date + 1e-6) if pv_to_date > 0 else 1.0
        eac = est_cost_input / (cpi + 1e-6) if cpi > 0 else est_cost_input
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("CPI", f"{cpi:.2f}")
        m2.metric("SPI", f"{spi:.2f}")
        m3.metric("EAC (Forecast)", f"${eac:,.2f}")
        m4.metric("STAGE PROGRESS", f"{prog*100:.1f}%")
        
        st.info(f"**Performance:** {'\u2705 Under Budget' if cpi >= 1 else '\ud83d\udd34 Over Budget'} | {'\u2705 Ahead' if spi >= 1 else '\ud83d\udd34 Behind Schedule'}")

    # Visual Breakdown Table
    st.markdown("### \ud83d\udcca Stage-by-Stage Baseline Breakdown")
    table_data = []
    for i, s in enumerate(STAGES):
        table_data.append({
            "Stage": s.replace('_', ' '),
            "AI Predicted Cost": f"${pred_costs[i]:,.2f}",
            "Distributed Planned Cost (PV)": f"${stages_data[s]['Cost']:,.2f}",
            "Planned Days": f"{stages_data[s]['Days']:.1f}"
        })
    st.table(pd.DataFrame(table_data))
