import argparse
import joblib
import numpy as np
import random
import sys

# --- CONSTANTS ---
STAGES = ['Site_Prep', 'Foundations', 'Structure', 'Systems', 'Finishing']
UNIT_COSTS = {
    'Building': 2500, 'Road': 500, 'Bridge': 5500,
    'Water Infra': 2200, 'Power Plant': 7500
}

three_point_ranges = {
    'Building': {
        'Time': [(5,7,10), (10,12,15), (15,20,25), (20,25,30), (25,30,35)],
        'Cost': [(3,5,7), (10,13,15), (20,25,30), (20,28,35), (15,20,25)],
        'Risk': [(40,45,50), (80,85,90), (30,38,45), (50,58,65), (20,28,35)]
    }
}

def get_p90_monte_carlo_weights(p_type, metric_key):
    # Simplified Monte Carlo for CLI
    data_map = three_point_ranges.get(p_type, three_point_ranges['Building'])
    tri_params = data_map.get(metric_key)
    iterations = 500
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

def run_analysis(p_type, area, complexity, resources):
    try:
        model = joblib.load('phi_model.pkl')
        scaler_x = joblib.load('scaler_x.pkl')
        scaler_y = joblib.load('scaler_y.pkl')
    except Exception as e:
        print(f"Error loading model files: {e}")
        sys.exit(1)

    # 1. Baseline Calculations
    base_cost = area * UNIT_COSTS.get(p_type, 1000) * (1 + (complexity-2)*0.2)
    base_days = (base_cost / 5000) * (1 + (complexity-2)*0.1)
    base_risk_for_stages = 50.0 + (complexity * 10)
    engineered_area = round(base_cost / UNIT_COSTS.get(p_type, 1000.0), 2)
    historical_cost_deviation = 5.0

    # 2. Stage Distributions
    c_w = get_p90_monte_carlo_weights(p_type, 'Cost')
    t_w = get_p90_monte_carlo_weights(p_type, 'Time')
    r_w = get_p90_monte_carlo_weights(p_type, 'Risk')

    planned_stages = []
    for i in range(len(STAGES)):
        planned_stages.append(base_cost * c_w[i])
        planned_stages.append(base_days * t_w[i])
        planned_stages.append(base_risk_for_stages * r_w[i])

    # 3. Dummies
    project_types_for_dummies = ['Bridge', 'Building', 'Power Plant', 'Road', 'Water Infra']
    type_dummies = [1.0 if t == p_type else 0.0 for t in project_types_for_dummies]

    # 4. Predict
    input_data = [
        engineered_area, float(complexity), float(resources),
        base_cost, historical_cost_deviation, base_days
    ] + planned_stages + type_dummies

    input_array = np.array(input_data).reshape(1, -1)
    input_scaled = scaler_x.transform(input_array)
    preds_scaled = model.predict(input_scaled)
    preds = scaler_y.inverse_transform(preds_scaled)[0]

    # Results
    predicted_duration = preds[0]
    predicted_cost = preds[2]
    
    # Calculate PHI
    norm_risk = (100 - base_risk_for_stages) / 100
    phi_score = (0.4 * 1.0) + (0.3 * 1.0) + (0.3 * norm_risk) # Simplified for CLI predictive mode
    phi_pct = phi_score * 100

    print(f"\n--- PHI ADVANCED CLI ANALYSIS ---")
    print(f"Project Type: {p_type}")
    print(f"Overall PHI:  {phi_pct:.1f}%")
    print(f"---------------------------------")
    print(f"AI Projected Cost:     ${predicted_cost:,.2f}")
    print(f"AI Projected Duration: {predicted_duration:,.1f} Days")
    print(f"Baseline Plan:         ${base_cost:,.2f} | {base_days:,.1f} Days")
    print(f"---------------------------------\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PHI Advanced CLI Tool")
    parser.add_argument("--type", type=str, default="Building", choices=list(UNIT_COSTS.keys()))
    parser.add_argument("--area", type=float, required=True)
    parser.add_argument("--complexity", type=int, choices=[1, 2, 3], default=2)
    parser.add_argument("--resources", type=int, default=85)

    args = parser.parse_args()
    run_analysis(args.type, args.area, args.complexity, args.resources)
