import argparse
import joblib
import numpy as np
import sys

def analyze_phi(area, complexity, resource_score, target_cost=0.0, target_schedule=0.0):
    try:
        model = joblib.load('phi_model.pkl')
        scaler_x = joblib.load('scaler_x.pkl')
        scaler_y = joblib.load('scaler_y.pkl')
    except Exception as e:
        print(f"Error loading model files: {e}")
        sys.exit(1)

    # 1. Prepare Input
    input_features = np.array([[area, complexity, resource_score]])
    input_scaled = scaler_x.transform(input_features)

    # 2. Predict
    preds_scaled = model.predict(input_scaled)
    preds = scaler_y.inverse_transform(preds_scaled)[0]

    # Model outputs from train_model.py: 
    # ['Historical_Cost_Deviation_%', 'Time_Estimate_Days', 'Risk_Assessment_Score', 'Total_Projected_Duration']
    pred_cost_dev = preds[0]
    pred_baseline_days = preds[1]
    pred_risk_score = preds[2]
    pred_total_duration = preds[3]

    # 3. PHI Logic (matching phi_tool.py)
    reference_days = target_schedule if target_schedule > 0 else pred_baseline_days
    estimated_overtime = max(0, pred_total_duration - reference_days)

    norm_cost = max(0, 1 - (pred_cost_dev / 25.0)) 
    norm_time = max(0, 1 - (estimated_overtime / reference_days))
    norm_risk = 1 - (pred_risk_score / 100.0)

    # Weights (Standard: 40/30/30)
    phi_score = (0.4 * norm_cost) + (0.3 * norm_time) + (0.3 * norm_risk)
    phi_score = np.clip(phi_score, 0, 1)

    # 4. Results
    status = "FEASIBLE" if phi_score > 0.75 else "BORDERLINE" if phi_score > 0.45 else "NOT FEASIBLE"
    
    print(f"\n--- PHI Analysis Result ---")
    print(f"PHI Score:      {phi_score:.2f} ({status})")
    print(f"Cost Health:    {norm_cost:.2f}")
    print(f"Time Health:    {norm_time:.2f}")
    print(f"Risk Health:    {norm_risk:.2f}")
    print(f"Projected Duration: {int(pred_total_duration)} days")
    print(f"---------------------------\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Project Health Indicator (PHI) CLI Tool")
    parser.add_argument("--area", type=float, required=True, help="Project Area (sqm)")
    parser.add_argument("--complexity", type=int, choices=[1, 2, 3], required=True, help="Complexity (1-3)")
    parser.add_argument("--resources", type=int, required=True, help="Resource Confidence (0-100)")
    parser.add_argument("--cost", type=float, default=0.0, help="Target Cost (USD)")
    parser.add_argument("--days", type=float, default=0.0, help="Target Schedule (Days)")

    args = parser.parse_args()
    analyze_phi(args.area, args.complexity, args.resources, args.cost, args.days)
