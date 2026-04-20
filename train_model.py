import pandas as pd
import numpy as np
import random
import re
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
import joblib

# 1. DATA LOADING & CLEANING
try:
    df_raw = pd.read_csv('Engineering_Cost_Feasibility_Dataset.csv')
except FileNotFoundError:
    print("Error: 'Engineering_Cost_Feasibility_Dataset.csv' not found.")
    exit(1)

def clean_numeric(value):
    if pd.isna(value): return 0.0
    clean_val = re.sub(r'[^\d.]', '', str(value))
    try: return float(clean_val)
    except: return 0.0

num_cols = ['Estimated_Cost_USD', 'Time_Estimate_Days', 'Resource_Allocation_Score', 'Historical_Cost_Deviation_%', 'Risk_Assessment_Score']
for col in num_cols:
    df_raw[col] = df_raw[col].apply(clean_numeric)

unit_cost_lookup = {'Building': 2500, 'Road': 500, 'Bridge': 5500, 'Water Infra': 2200, 'Power Plant': 7500}

# 2. DATA AUGMENTATION LOGIC
stages = ['Site_Prep', 'Foundations', 'Structure', 'Systems', 'Finishing']
ranges = {
    'Building': {
        'Time': [(5, 10), (10, 15), (15, 25), (20, 30), (25, 35)],
        'Cost': [(3, 7), (10, 15), (20, 30), (20, 35), (15, 25)],
        'Risk': [(40, 50), (80, 90), (30, 45), (50, 65), (20, 35)]
    }
}

def get_stage_values(p_type, metric, base_target):
    r = ranges.get(p_type, ranges['Building'])
    m_r = r.get(metric, r['Time'])
    vals = [random.uniform(l, h) for (l, h) in m_r]
    if metric == 'Risk':
        target = base_target * 5
        diff = (target - sum(vals)) / 5
        return [round(max(l, min(h, v + diff)), 2) for i, (v, (l, h)) in enumerate(zip(vals, m_r))]
    else:
        total = sum(vals)
        return [round((v / total) * base_target, 2) for v in vals]

def process_augmented_data(base_df, target_rows=21000):
    expanded = []
    multiplier = max(1, target_rows // len(base_df))
    for _ in range(multiplier):
        for _, original_row in base_df.iterrows():
            row = original_row.to_dict()
            p_type = str(row['Project_Type'])
            base_cost, base_time, base_risk = float(row['Estimated_Cost_USD']), float(row['Time_Estimate_Days']), float(row['Risk_Assessment_Score'])
            
            costs = get_stage_values(p_type, 'Cost', base_cost)
            days = get_stage_values(p_type, 'Time', base_time)
            risks = get_stage_values(p_type, 'Risk', base_risk)
            
            for i, stage in enumerate(stages):
                row[f'{stage}_Cost'], row[f'{stage}_Days'], row[f'{stage}_Risk'] = costs[i], days[i], risks[i]
            
            row['Finishing_Cost'] += round(base_cost - sum(costs), 2)
            row['Finishing_Days'] += round(base_time - sum(days), 2)
            row['Finishing_Risk'] += round(base_risk * 5 - sum(risks), 2)
            
            ot_days = (base_time * (base_risk/100) * 0.3) + (base_time * (1 - row['Resource_Allocation_Score']/100) * 0.2)
            row['Predicted_Final_Duration'] = round(base_time + ot_days, 2)
            row['Predicted_Delay_Days'] = round(ot_days, 2)
            row['Predicted_Final_Cost'] = round(base_cost + (base_cost * row['Historical_Cost_Deviation_%']/100) + (ot_days * base_cost * 0.001), 2)
            row['Engineered_Area'] = round(base_cost / unit_cost_lookup.get(p_type, 1000), 2)
            if 'Scope_Complexity_Numeric' not in row: row['Scope_Complexity_Numeric'] = 2
            expanded.append(row)
    return pd.DataFrame(expanded)

print("Preparing data...")
df_processed = process_augmented_data(df_raw)
df_ml = pd.get_dummies(df_processed, columns=['Project_Type'])

# Define EXPLICIT UNIQUE features
base_features = ['Engineered_Area', 'Scope_Complexity_Numeric', 'Resource_Allocation_Score', 'Estimated_Cost_USD', 'Historical_Cost_Deviation_%', 'Time_Estimate_Days', 'Risk_Assessment_Score']
stage_features = []
for s in stages:
    stage_features += [f'{s}_Cost', f'{s}_Days', f'{s}_Risk']
type_features = [col for col in df_ml.columns if 'Project_Type_' in col]

features = base_features + stage_features + type_features
targets = ['Predicted_Final_Duration', 'Predicted_Delay_Days', 'Predicted_Final_Cost']

X = df_ml[features].values.astype(float)
y = df_ml[targets].values.astype(float)

scaler_x, scaler_y = MinMaxScaler(), MinMaxScaler()
X_scaled = scaler_x.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

print(f"Training on {len(features)} unique features...")
model = MLPRegressor(hidden_layer_sizes=(128, 128), max_iter=1000, random_state=42)
model.fit(X_train, y_train)

joblib.dump(model, 'phi_model.pkl')
joblib.dump(scaler_x, 'scaler_x.pkl')
joblib.dump(scaler_y, 'scaler_y.pkl')
joblib.dump(features, 'feature_names.pkl') # Save feature names for reference

print(f"Complete. R2: {model.score(X_test, y_test):.4f}")
