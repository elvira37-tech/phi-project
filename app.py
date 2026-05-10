import streamlit as st
import pandas as pd
import joblib
import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx

# --- 1. JAX Model Architecture ---
class ConstructionNN(nnx.Module):
    def __init__(self, num_params, num_output, rngs: nnx.Rngs):
        self.linear1 = nnx.Linear(num_params, 64, rngs=rngs)
        self.linear2 = nnx.Linear(64, 64, rngs=rngs)
        self.linear3 = nnx.Linear(64, num_output, rngs=rngs)

    def __call__(self, x):
        x = nnx.tanh(self.linear1(x))
        x = nnx.tanh(self.linear2(x))
        return self.linear3(x)

# --- 2. Load Assets ---
@st.cache_resource
def load_assets():
    import os
    try:
        model_path = 'jax_project_model.pkl'
        if not os.path.exists(model_path):
            st.error(f"Model file not found at {os.path.abspath(model_path)}")
            return None, None
            
        assets = joblib.load(model_path)
        model = ConstructionNN(len(assets['param_cols']), len(assets['output_cols']), rngs=nnx.Rngs(0))
        nnx.update(model, assets['model_state'])
        return model, assets
    except Exception as e:
        import traceback
        st.error(f"Error loading model: {str(e)}")
        st.code(traceback.format_exc())
        return None, None

model, assets = load_assets()

# --- 3. Streamlit UI ---
st.set_page_config(page_title="PHI - Project Intelligence", layout="wide")
st.title("🏗️ PHI - Project Intelligence Dashboard")

if not assets:
    st.info("👋 Welcome! Please run `python train_model.py` to generate the model assets before using the dashboard.")
    st.stop()

st.sidebar.header("1. Project Parameters")
p_type = st.sidebar.selectbox("Project Type", ['Building', 'Power Plant', 'Road', 'Bridge', 'Water Infra'])
area = st.sidebar.number_input("Engineered Area (sqm)", value=5000.0, step=100.0)
est_cost_input = st.sidebar.number_input("Estimated Total Cost (USD)", value=1000000.0, step=10000.0)
est_days_input = st.sidebar.number_input("Estimated Total Time (Days)", value=300.0, step=1.0)

st.sidebar.header("2. Complexity & Risk")
complexity_map = {'Low': 1, 'Medium': 2, 'High': 3}
complexity_label = st.sidebar.select_slider("Complexity", options=['Low', 'Medium', 'High'], value='Medium')
complexity = complexity_map[complexity_label]
risk_score = st.sidebar.slider("Initial Risk Score", 0, 100, 50)
resource_score = st.sidebar.slider("Resource Score", 0, 100, 75)

mode = st.sidebar.radio("Operation Mode", ["Estimate Project", "Tracking Project"])

STAGES = ['Site_Prep', 'Foundations', 'Structure', 'Systems', 'Finishing']

# --- Tracking UI ---
if mode == "Tracking Project":
    st.subheader("🔄 Real-Time Stage Tracking")
    c1, c2, c3 = st.columns(3)
    with c1: curr_stage = st.selectbox("Current Stage", STAGES)
    with c2: act_cost = st.number_input("Actual Cost to Date (USD)", value=0.0, step=1000.0)
    with c3: act_days = st.number_input("Actual Days Spent in Stage", value=0.0, step=1.0)

if st.button("🚀 RUN ANALYSIS", use_container_width=True):
    # 4. Prepare Input
    param_cols = assets['param_cols']
    input_vector_dict = {
        'Engineered_Area': float(area),
        'Scope_Complexity_Numeric': float(complexity),
        'Risk_Assessment_Score': float(risk_score),
        'Resource_Allocation_Score': float(resource_score),
        'Time_Estimate_Days': float(est_days_input)
    }

    # Initialize one-hot encoded project type columns to 0
    for col in param_cols:
        if col.startswith('Proj_'):
            input_vector_dict[col] = 0.0

    # Set the selected project type to 1
    selected_proj_label = p_type.replace(' ', '_')
    selected_proj_col = f"Proj_{selected_proj_label}"
    if selected_proj_col in input_vector_dict:
        input_vector_dict[selected_proj_col] = 1.0
    else:
        # Fallback in case of naming mismatch (e.g., 'Water_Infra' vs 'Water_Infrastructure')
        for col in param_cols:
            if col.startswith('Proj_') and selected_proj_label in col:
                input_vector_dict[col] = 1.0
                break

    # Create ordered input array
    ordered_input = np.array([input_vector_dict[col] for col in param_cols], dtype=np.float32)

    # Scale
    X_scaled = (ordered_input - assets['params_min']) / (assets['params_range'] + 1e-6)
    
    # Inference
    y_pred_norm = model(jnp.array(X_scaled.reshape(1, -1), dtype=jnp.float32))
    y_pred = np.array(y_pred_norm) * assets['output_range'] + assets['output_min']

    # Map Stage Predictions
    pred_costs = np.maximum(y_pred[0, 0:5], 0.0)
    pred_times = np.maximum(y_pred[0, 5:10], 0.0)

    ai_total_cost = sum(pred_costs)
    ai_total_days = sum(pred_times)

    # Distribute based on proportions
    stages_data = {}
    for i, s in enumerate(STAGES):
        cost_weight = pred_costs[i] / (ai_total_cost + 1e-6)
        days_weight = pred_times[i] / (ai_total_days + 1e-6)
        
        stages_data[s] = {
            'PV_C': est_cost_input * cost_weight,
            'PV_D': est_days_input * days_weight
        }

    if mode == "Estimate Project":
        st.subheader("📊 Project Estimation")
        m1, m2 = st.columns(2)
        m1.metric("BUDGET BASELINE (PV)", f"${est_cost_input:,.2f}",
                  delta=f"${(est_cost_input - ai_total_cost):,.0f} vs AI Model Prediction")
        m2.metric("DURATION BASELINE", f"{int(est_days_input)} Days",
                  delta=f"{int(est_days_input - ai_total_days)} Days vs AI Model Prediction")

    else:
        # EVM Logic
        idx = STAGES.index(curr_stage)
        pv_to_date = sum(stages_data[s]['PV_C'] for s in STAGES[:idx+1])
        prev_ev = sum(stages_data[s]['PV_C'] for s in STAGES[:idx])
        prog = min(1.0, act_days / (stages_data[curr_stage]['PV_D'] + 1e-6))
        ev = prev_ev + (stages_data[curr_stage]['PV_C'] * prog)
        
        cpi = ev / (act_cost + 1e-6) if act_cost > 0 else 1.0
        spi = ev / (pv_to_date + 1e-6) if pv_to_date > 0 else 1.0
        eac = est_cost_input / (cpi + 1e-6)

        st.subheader("📈 Performance Metrics")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("CPI", f"{cpi:.2f}", help="Cost Performance Index")
        m2.metric("SPI", f"{spi:.2f}", help="Schedule Performance Index")
        m3.metric("EAC (Forecast)", f"${eac:,.2f}")
        m4.metric("STAGE PROGRESS", f"{prog*100:.1f}%")

        color = "green" if cpi >= 1 and spi >= 1 else "orange" if cpi >= 1 or spi >= 1 else "red"
        st.markdown(f"**Performance Status:** <span style='color:{color}'>{'🟢 Healthy' if color=='green' else '🟡 Warning' if color=='orange' else '🔴 Critical'}</span>", unsafe_allow_html=True)

    # Breakdown Table
    st.markdown("### 📊 Stage-by-Stage Baseline Breakdown")
    table_data = []
    for i, s in enumerate(STAGES):
        table_data.append({
            "Stage": s.replace('_', ' '),
            "AI Prediction (Cost)": f"${pred_costs[i]:,.2f}",
            "Distributed Budget (PV)": f"${stages_data[s]['PV_C']:,.2f}",
            "Planned Days": f"{stages_data[s]['PV_D']:.1f}"
        })
    st.table(pd.DataFrame(table_data))
