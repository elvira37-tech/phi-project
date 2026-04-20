import streamlit as st
import pandas as pd
import joblib
import numpy as np
import random

# --- 1. APP CONFIGURATION ---
st.set_page_config(page_title="PHI - Project Health Suite", layout="wide")

# --- 2. ASSET LOADING ---
@st.cache_resource
def load_assets():
    try:
        model = joblib.load('phi_model.pkl')
        scaler_x = joblib.load('scaler_x.pkl')
        scaler_y = joblib.load('scaler_y.pkl')
        return model, scaler_x, scaler_y
    except Exception as e:
        st.error(f"Error loading AI assets: {e}. Please ensure 'phi_model.pkl', 'scaler_x.pkl', and 'scaler_y.pkl' are in the same directory.")
        return None, None, None

model, scaler_x, scaler_y = load_assets()

# --- 3. DATA CONSTANTS ---
STAGES = ['Site_Prep', 'Foundations', 'Structure', 'Systems', 'Finishing']
# Aligning with Project Types used in model training
UNIT_COSTS = {
    'Building': 2500, 'Road': 500, 'Bridge': 5500,
    'Water Infra': 2200, 'Power Plant': 7500
}

# Consistent with notebook's three_point_ranges
three_point_ranges = {
    'Building': {
        'Time': [(5,7,10), (10,12,15), (15,20,25), (20,25,30), (25,30,35)],
        'Cost': [(3,5,7), (10,13,15), (20,25,30), (20,28,35), (15,20,25)],
        'Risk': [(40,45,50), (80,85,90), (30,38,45), (50,58,65), (20,28,35)]
    },
    'Road': {
        'Time': [(10,15,20), (5,8,10), (40,45,50), (10,12,15), (5,7,10)],
        'Cost': [(15,20,25), (10,12,15), (30,40,45), (10,13,15), (5,7,10)],
        'Risk': [(20,25,30), (10,15,20), (40,45,50), (10,13,15), (5,7,10)]
    },
    'Bridge': {
        'Time': [(5,10,15), (20,25,30), (30,40,45), (10,12,15), (5,8,10)],
        'Cost': [(5,8,10), (25,30,35), (35,40,45), (10,13,15), (5,7,10)],
        'Risk': [(10,15,20), (40,50,60), (30,35,40), (10,15,20), (5,8,10)]
    },
    'Water Infra': {
        'Time': [(10,12,15), (20,22,25), (25,30,35), (15,18,20), (10,12,15)],
        'Cost': [(10,13,15), (15,20,25), (30,35,40), (15,18,20), (5,8,10)],
        'Risk': [(20,25,30), (20,25,30), (20,25,30), (10,13,15), (5,8,10)]
    },
    'Power Plant': {
        'Time': [(5,8,10), (15,18,20), (35,40,45), (20,22,25), (10,12,15)],
        'Cost': [(5,7,10), (10,15,20), (40,45,50), (20,23,25), (5,7,10)],
        'Risk': [(10,12,15), (20,25,30), (40,45,50), (15,18,20), (5,8,10)]
    }
}

def get_p90_monte_carlo_weights(p_type, metric_key, base_target=None):
    """
    Runs Monte Carlo to find P90 weights.
    Cost/Time = Sum Logic | Risk = Average Logic
    """
    data_map = three_point_ranges.get(p_type, three_point_ranges['Building'])
    tri_params = data_map.get(metric_key)
    iterations = 1000
    sims = []

    for _ in range(iterations):
        sample = [random.triangular(a, m, b) for (a, m, b) in tri_params]

        if metric_key == 'Risk':
            avg_sample = sum(sample) / len(sample)
            sims.append([s / (avg_sample / 1.0) for s in sample])
        else:
            total = sum(sample)
            sims.append([s / total for s in sample])

    p90_raw = np.percentile(np.array(sims), 90, axis=0)

    if metric_key == 'Risk':
        return p90_raw / (sum(p90_raw) / len(p90_raw))
    else:
        return p90_raw / sum(p90_raw)


# --- 4. SIDEBAR WEIGHTING ---
st.sidebar.header("⚖️ PHI Weighting Configuration")
w_cost_val = st.sidebar.slider("Cost Weight (%)", 0, 100, 40)
w_time_val = st.sidebar.slider("Time Weight (%)", 0, 100, 30)
w_risk_val = st.sidebar.slider("Risk Weight (%)", 0, 100, 30)

# Normalize weights to sum to 1.0
total_w = w_cost_val + w_time_val + w_risk_val
if total_w == 0:
    w_cost, w_time, w_risk = 0.33, 0.33, 0.34
else:
    w_cost = w_cost_val / total_w
    w_time = w_time_val / total_w
    w_risk = w_risk_val / total_w

# --- 5. MANDATORY & OPTIONAL INPUTS ---
st.title("🏗️ PHI - Project Health Index Suite")
st.markdown("---")

st.subheader("📋 Core Project Parameters")
col1, col2, col3 = st.columns(3)
with col1:
    p_type = st.selectbox("Project Type*", list(UNIT_COSTS.keys()))
with col2:
    area = st.number_input("Project Area (sqm)*", min_value=1.0, value=1200.0)
with col3:
    complexity = st.slider("Complexity (1-3)*", 1, 3, 2)

with st.expander("➕ Optional Initial Estimates"):
    opt_c1, opt_c2, opt_c3 = st.columns(3)
    # These target values are not directly used by the model but can be used for reference in the UI
    target_cost = opt_c1.number_input("Original Cost Estimate (USD)", value=0.0)
    target_days = opt_c2.number_input("Original Schedule Estimate (Days)", value=0.0)
    res_score = opt_c3.slider("Resource Confidence (%)", 0, 100, 85)
    historical_cost_deviation = 5.0 # Hardcoded for now, could be a slider

# --- 6. DYNAMIC MULTI-STAGE TRACKING ---
st.markdown("---")
st.subheader("🔄 Project Progress Tracking")
status = st.radio("Current Project Status:", ["No - Predictive Mode", "Yes - Tracking Mode"])

actual_stages_data = []

if status == "Yes - Tracking Mode":
    st.info("Select all completed stages and enter their specific actual data.")
    completed_stages = st.multiselect("Select Completed Stages:", STAGES)

    for stage in completed_stages:
        with st.container():
            st.markdown(f"**Data Entry for: {stage.replace('_', ' ')}**")
            sc1, sc2, sc3 = st.columns(3)
            a_cost = sc1.number_input(f"Actual Cost ({stage})", min_value=0.0, key=f"c_{stage}")
            a_days = sc2.number_input(f"Actual Days ({stage})", min_value=0.0, key=f"d_{stage}")
            a_risk = sc3.slider(f"Observed Risk ({stage})", 0, 100, 40, key=f"r_{stage}")

            actual_stages_data.append({
                "Stage": stage,
                "Cost": a_cost,
                "Days": a_days,
                "Risk": a_risk
            })
            st.markdown("---")

# --- 7. LOGIC & CALCULATIONS ---
if st.button("🚀 RUN PHI ANALYSIS"):
    if model is None or scaler_x is None or scaler_y is None:
        st.error("AI assets not loaded. Check the errors above.")
    else:
        # 1. Baseline Calculations
        base_cost = area * UNIT_COSTS.get(p_type, 1000) * (1 + (complexity-2)*0.2)
        base_days = (base_cost / 5000) * (1 + (complexity-2)*0.1)
        # base_risk_for_stages acts as the 'Risk_Assessment_Score' used in training
        base_risk_for_stages = 50.0 + (complexity * 10) 
        engineered_area = round(base_cost / UNIT_COSTS.get(p_type, 1000.0), 2)

        # 2. Distribution Weights
        c_w = get_p90_monte_carlo_weights(p_type, 'Cost')
        t_w = get_p90_monte_carlo_weights(p_type, 'Time')
        r_w = get_p90_monte_carlo_weights(p_type, 'Risk')

        # 3. Construct Feature Vector (MATCHES TRAIN_MODEL.PY EXACTLY)
        planned_stages = []
        for i in range(len(STAGES)):
            planned_stages.extend([
                base_cost * c_w[i], 
                base_days * t_w[i], 
                base_risk_for_stages * r_w[i]
            ])

        project_types = ['Bridge', 'Building', 'Power Plant', 'Road', 'Water Infra']
        type_dummies = [1.0 if t == p_type else 0.0 for t in project_types]

        # Final input list: 7 base features + 15 stage features + 5 type dummies = 27 total features
        input_data = [
            engineered_area,               # 1
            float(complexity),             # 2
            float(res_score),              # 3
            base_cost,                     # 4
            historical_cost_deviation,     # 5
            base_days,                     # 6
            base_risk_for_stages           # 7 (THE MISSING FEATURE)
        ] + planned_stages + type_dummies

        # 4. Prediction
        input_scaled = scaler_x.transform(np.array(input_data).reshape(1, -1))
        preds_scaled = model.predict(input_scaled)
        preds = scaler_y.inverse_transform(preds_scaled)[0]

        # Model Target Mapping: [Duration, Delay, Cost]
        ai_final_days = preds[0]
        ai_final_cost = preds[2]

        # 5. Tracking vs Predictive Logic
        if status == "Yes - Tracking Mode" and actual_stages_data:
            total_actual_cost = sum(i['Cost'] for i in actual_stages_data)
            total_actual_days = sum(i['Days'] for i in actual_stages_data)
            avg_actual_risk = sum(i['Risk'] for i in actual_stages_data) / len(actual_stages_data)
            
            progress_pct = (STAGES.index(actual_stages_data[-1]['Stage']) + 1) / len(STAGES)
            final_eac_cost = total_actual_cost + (ai_final_cost * (1 - progress_pct))
            final_eac_days = total_actual_days + (ai_final_days * (1 - progress_pct))
            
            cpi = (base_cost * progress_pct) / total_actual_cost if total_actual_cost > 0 else 1.0
            spi = (base_days * progress_pct) / total_actual_days if total_actual_days > 0 else 1.0
            s_risk = avg_actual_risk
        else:
            final_eac_cost, final_eac_days = ai_final_cost, ai_final_days
            cpi, spi, s_risk = 1.0, 1.0, base_risk_for_stages

        # 6. PHI Calculation
        norm_cpi = np.clip(cpi, 0, 1.2)
        norm_spi = np.clip(spi, 0, 1.2)
        norm_risk = (100 - np.clip(s_risk, 0, 100)) / 100
        phi_index = (norm_cpi * 85 * w_cost) + (norm_spi * 85 * w_time) + (norm_risk * 100 * w_risk)

        # --- 8. DASHBOARD DISPLAY ---
        st.divider()
        st.markdown(f"""
            <div style="background-color: #ffffff; border: 2px solid #b71c1c; padding: 25px; border-radius: 15px; text-align: center;">
                <h1 style="color: #b71c1c; font-size: 50px; margin: 0;">{phi_index:.1f}%</h1>
                <p style="color: #666; font-weight: bold;">PROJECT HEALTH INDEX</p>
            </div>
        """, unsafe_allow_html=True)

        m1, m2, m3 = st.columns(3)
        m1.metric("Cost Performance (CPI)", f"{cpi:.2f}")
        m2.metric("Schedule Performance (SPI)", f"{spi:.2f}")
        m3.metric("Risk Level", f"{s_risk:.0f}/100")

        st.subheader("📋 Detailed Stage Forecast")
        breakdown = []
        for i, s in enumerate(STAGES):
            breakdown.append({
                "Stage": s.replace('_', ' '),
                "Est. Cost": f"${ai_final_cost * c_w[i]:,.2f}",
                "Est. Days": f"{ai_final_days * t_w[i]:,.1f}",
                "Risk": f"{base_risk_for_stages * r_w[i]:.1f}%"
            })
        st.table(pd.DataFrame(breakdown))
        
        st.success(f"**Estimated Final Cost:** ${final_eac_cost:,.2f}")
        st.info(f"**Estimated Final Duration:** {final_eac_days:,.1f} Days")