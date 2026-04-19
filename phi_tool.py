import streamlit as st
import pandas as pd
import joblib
import numpy as np

# 1. LOAD MODELS & SCALERS
try:
    model = joblib.load('phi_model.pkl')
    scaler_x = joblib.load('scaler_x.pkl')
    scaler_y = joblib.load('scaler_y.pkl')
except Exception as e:
    st.error(f"Error loading model files: {e}")

st.set_page_config(page_title="PHI - Project Health Index", layout="wide")
st.title("🏗️ Project Health Index (PHI) Tool")

# 2. SIDEBAR: WEIGHTING CONFIGURATION
st.sidebar.header("⚖️ Weighting Configuration")
weight_mode = st.sidebar.radio("Weighting Strategy", ["Standard (Default)", "Custom Weights"])

if weight_mode == "Standard (Default)":
    w_cost, w_time, w_risk = 0.4, 0.3, 0.3
    st.sidebar.info("Standard Weights: \n- Cost: 40% \n- Time: 30% \n- Risk: 30%")
else:
    st.sidebar.subheader("Adjust Weights")
    w_cost_val = st.sidebar.slider("Cost Weight (%)", 0, 100, 40)
    w_time_val = st.sidebar.slider("Time/Overtime Weight (%)", 0, 100, 30)
    w_risk_val = st.sidebar.slider("Risk Weight (%)", 0, 100, 30)

    # Logic to ensure total is 100%
    total_w = w_cost_val + w_time_val + w_risk_val
    if total_w == 0:
        w_cost, w_time, w_risk = 0.33, 0.33, 0.34
    else:
        # Normalize to ensure sum is exactly 1.0
        w_cost = w_cost_val / total_w
        w_time = w_time_val / total_w
        w_risk = w_risk_val / total_w

    st.sidebar.write(f"**Applied Ratios:** Cost: {w_cost:.2f} | Time: {w_time:.2f} | Risk: {w_risk:.2f}")
    if total_w != 100:
        st.sidebar.warning(f"Weights adjusted proportionally to sum to 100% (Current sum: {total_w}%)")

# 3. INPUT SECTION
st.header("📋 Project Parameters")
col1, col2, col3 = st.columns(3)
with col1:
    p_type = st.selectbox("Project Type*",
        ['Building', 'Industrial Complex', 'Road', 'Bridge', 'Water Infra', 'Smart Solar Grid', 'Urban Flyover', 'Dam Reinforcement'])
with col2:
    area = st.number_input("Project Area (Units/sqm)*", min_value=1.0, value=500.0)
with col3:
    complexity = st.slider("Complexity (1-3)*", 1, 3, 2)

with st.expander("➕ Optional Estimates & Resources"):
    opt_col1, opt_col2, opt_col3 = st.columns(3)
    with opt_col1:
        user_cost = st.number_input("Target Cost (USD)", value=0.0)
    with opt_col2:
        user_schedule = st.number_input("Target Schedule (Days)", value=0.0)
    with opt_col3:
        resource_score = st.slider("Resource Confidence (%)", 0, 100, 80)

# 4. PREDICTION & CALCULATION
if st.button("🚀 Calculate Project Health"):
    input_features = np.array([[area, complexity, resource_score]])
    input_scaled = scaler_x.transform(input_features)

    preds_scaled = model.predict(input_scaled)
    preds = scaler_y.inverse_transform(preds_scaled)[0]

    pred_cost_dev = preds[0]
    pred_baseline_days = preds[1]
    pred_risk_score = preds[2]
    pred_total_duration = preds[3]

    # Dynamic Overtime Calculation
    reference_days = user_schedule if user_schedule > 0 else pred_baseline_days
    estimated_overtime = max(0, pred_total_duration - reference_days)

    # Normalization (0 to 1)
    norm_cost = max(0, 1 - (pred_cost_dev / 25.0)) # 25% dev = 0 score
    norm_time = max(0, 1 - (estimated_overtime / reference_days))
    norm_risk = 1 - (pred_risk_score / 100.0)

    # 5. PHI FORMULA WITH VARIABLE WEIGHTS
    phi_score = (w_cost * norm_cost) + (w_time * norm_time) + (w_risk * norm_risk)
    phi_score = np.clip(phi_score, 0, 1)

    # 6. OUTPUT DISPLAY
    st.divider()
    res_col1, res_col2 = st.columns([1, 2])
    with res_col1:
        st.metric("Overall PHI Score", f"{phi_score:.2f}")
    with res_col2:
        if phi_score > 0.75: st.success("STATUS: FEASIBLE")
        elif phi_score > 0.45: st.warning("STATUS: BORDERLINE")
        else: st.error("STATUS: NOT FEASIBLE")

    m1, m2, m3 = st.columns(3)
    m1.metric("Cost Health", f"{norm_cost:.2f}", help=f"Cost Dev: {pred_cost_dev:.1f}%")
    m2.metric("Time Health", f"{norm_time:.2f}", help=f"Overtime: {int(estimated_overtime)} Days")
    m3.metric("Risk Health", f"{norm_risk:.2f}", help=f"Risk Score: {pred_risk_score:.1f}")

    st.info(f"Using **{weight_mode}** strategy. Total projected duration: **{int(pred_total_duration)} days**.")
