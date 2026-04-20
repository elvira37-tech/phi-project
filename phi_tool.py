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
        st.error(f"Error loading AI assets: {e}")
        return None, None, None

model, scaler_x, scaler_y = load_assets()

# --- 3. DATA CONSTANTS ---
STAGES = ['Site_Prep', 'Foundations', 'Structure', 'Systems', 'Finishing']
UNIT_COSTS = {'Building': 2500, 'Road': 500, 'Bridge': 5500, 'Water Infra': 2200, 'Power Plant': 7500}
PROJECT_TYPES = ['Bridge', 'Building', 'Power Plant', 'Road', 'Water Infra']

three_point_ranges = {
    'Building': {
        'Time': [(5,7,10), (10,12,15), (15,20,25), (20,25,30), (25,30,35)],
        'Cost': [(3,5,7), (10,13,15), (20,25,30), (20,28,35), (15,20,25)],
        'Risk': [(40,45,50), (80,85,90), (30,38,45), (50,58,65), (20,28,35)]
    }
}

def get_p90_monte_carlo_weights(p_type, metric_key):
    r = three_point_ranges.get(p_type, three_point_ranges['Building'])
    tri_params = r.get(metric_key)
    sims = []
    for _ in range(1000):
        sample = [random.triangular(a, m, b) for (a, m, b) in tri_params]
        if metric_key == 'Risk':
            avg = sum(sample) / len(sample)
            sims.append([s / avg for s in sample])
        else:
            total = sum(sample)
            sims.append([s / total for s in sample])
    p90 = np.percentile(np.array(sims), 90, axis=0)
    return p90 / (sum(p90)/len(p90)) if metric_key == 'Risk' else p90 / sum(p90)

# --- 4. SIDEBAR ---
st.sidebar.header("⚖️ PHI Weighting")
w_cost_val = st.sidebar.slider("Cost Weight", 0, 100, 40)
w_time_val = st.sidebar.slider("Time Weight", 0, 100, 30)
w_risk_val = st.sidebar.slider("Risk Weight", 0, 100, 30)
total_w = w_cost_val + w_time_val + w_risk_val
w_cost, w_time, w_risk = (w_cost_val/total_w, w_time_val/total_w, w_risk_val/total_w) if total_w > 0 else (0.33, 0.33, 0.34)

# --- 5. INPUTS ---
st.title("🏗️ PHI - Project Health Index Suite")
st.markdown("---")
st.subheader("📋 Core Project Parameters")
col1, col2, col3 = st.columns(3)
with col1: p_type = st.selectbox("Project Type*", PROJECT_TYPES)
with col2: area = st.number_input("Project Area (sqm)*", min_value=1.0, value=1200.0)
with col3: complexity = st.slider("Complexity (1-3)*", 1, 3, 2)

with st.expander("➕ Optional Initial Estimates"):
    opt_c1, opt_c2, opt_c3 = st.columns(3)
    res_score = opt_c3.slider("Resource Confidence (%)", 0, 100, 85)
    hist_dev = 5.0

# --- 6. TRACKING ---
st.markdown("---")
st.subheader("🔄 Project Progress Tracking")
status = st.radio("Current Project Status:", ["No - Predictive Mode", "Yes - Tracking Mode"])
actual_stages_data = []
if status == "Yes - Tracking Mode":
    completed_stages = st.multiselect("Select Completed Stages:", STAGES)
    for stage in completed_stages:
        with st.container():
            st.markdown(f"**Data for: {stage.replace('_', ' ')}**")
            sc1, sc2, sc3 = st.columns(3)
            actual_stages_data.append({"Stage": stage, "Cost": sc1.number_input(f"Actual Cost ({stage})", 0.0, key=f"c_{stage}"), "Days": sc2.number_input(f"Actual Days ({stage})", 0.0, key=f"d_{stage}"), "Risk": sc3.slider(f"Observed Risk ({stage})", 0, 100, 40, key=f"r_{stage}")})

# --- 7. CALCULATIONS ---
if st.button("🚀 RUN PHI ANALYSIS"):
    if model:
        # Baseline Logic
        base_cost = area * UNIT_COSTS.get(p_type, 1000) * (1 + (complexity-2)*0.2)
        base_days = (base_cost / 5000) * (1 + (complexity-2)*0.1)
        base_risk = 50.0 + (complexity * 10)
        eng_area = round(base_cost / UNIT_COSTS.get(p_type, 1000.0), 2)
        
        # P90 Stage Weights
        c_w, t_w, r_w = get_p90_monte_carlo_weights(p_type, 'Cost'), get_p90_monte_carlo_weights(p_type, 'Time'), get_p90_monte_carlo_weights(p_type, 'Risk')
        
        # Build Feature Vector (27 features)
        # 1-7: Base Features
        input_data = [eng_area, float(complexity), float(res_score), base_cost, hist_dev, base_days, base_risk]
        # 8-22: Stage Features
        for i in range(5):
            input_data += [base_cost * c_w[i], base_days * t_w[i], base_risk * r_w[i]]
        # 23-27: Project Type Dummies
        input_data += [1.0 if t == p_type else 0.0 for t in PROJECT_TYPES]
        
        input_scaled = scaler_x.transform(np.array(input_data).reshape(1, -1))
        preds = scaler_y.inverse_transform(model.predict(input_scaled))[0]
        
        pred_duration, pred_delay, pred_cost = preds[0], preds[1], preds[2]
        
        # EAC & PHI Logic
        final_eac_cost, final_eac_days, s_risk, cpi, spi = pred_cost, pred_duration, base_risk, 1.0, 1.0
        if status == "Yes - Tracking Mode" and actual_stages_data:
            total_c, total_d = sum(i['Cost'] for i in actual_stages_data), sum(i['Days'] for i in actual_stages_data)
            progress = (STAGES.index(actual_stages_data[-1]['Stage']) + 1) / 5
            final_eac_cost = total_c + (pred_cost * (1-progress)) if progress < 1 else total_c
            final_eac_days = total_d + (pred_duration * (1-progress)) if progress < 1 else total_d
            cpi, spi = (base_cost * progress) / total_c if total_c > 0 else 1.0, (base_days * progress) / total_d if total_d > 0 else 1.0
            s_risk = sum(i['Risk'] for i in actual_stages_data) / len(actual_stages_data)
        
        phi_index = (np.clip(cpi,0,1.2)*85*w_cost) + (np.clip(spi,0,1.2)*85*w_time) + (((100-np.clip(s_risk,0,100))/100)*100*w_risk)

        # --- 8. DASHBOARD ---
        st.divider()
        st.markdown(f'<div style="background-color: #ffffff; border: 2px solid #b71c1c; padding: 25px; border-radius: 15px; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.1);"><h1 style="color: #b71c1c; font-size: 50px; margin: 0;">{phi_index:.1f}%</h1><p style="color: #666; font-weight: bold; text-transform: uppercase; letter-spacing: 1px;">Overall Project Health Index</p></div>', unsafe_allow_html=True)
        st.write("###")
        m1, m2, m3 = st.columns(3)
        m1.metric("CPI (Cost Index)", f"{cpi:.2f}", f"{cpi-1:.2f}")
        m2.metric("SPI (Schedule Index)", f"{spi:.2f}", f"{spi-1:.2f}")
        m3.metric("Aggregated Risk", f"{s_risk:.0f}/100", delta_color="inverse")
        
        st.subheader("📋 AI Forecast per Stage")
        breakdown = [{"Stage": s.replace('_',' '), "Predicted Cost": f"${pred_cost*c_w[i]:,.2f}", "Predicted Duration (Days)": f"{pred_duration*t_w[i]:,.1f}", "AI Risk Prob.": f"{base_risk*r_w[i]:.1f}%"} for i,s in enumerate(STAGES)]
        st.table(pd.DataFrame(breakdown))
        
        r1, r2 = st.columns(2)
        r1.success(f"**Final Predicted Cost (EAC):** ${final_eac_cost:,.2f}")
        r2.info(f"**Final Predicted Duration:** {final_eac_days:,.1f} Days")
