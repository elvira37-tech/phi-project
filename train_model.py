import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
import joblib

# ==========================================
# 1. DATA INGESTION & PRE-PROCESSING
# ==========================================
# Step 1: Load Dataset
df = pd.read_csv('Engineering_Cost_Feasibility_Dataset.csv')

# Step 2: Area Engineering
unit_cost_lookup = {
    'Building': 2500, 'Industrial Complex': 3500, 'Road': 500,
    'Bridge': 5500, 'Water Infra': 1200, 'Smart Solar Grid': 1500,
    'Urban Flyover': 4500, 'Dam Reinforcement': 7500
}

df['Engineered_Area'] = df.apply(
    lambda row: row['Estimated_Cost_USD'] / unit_cost_lookup.get(row['Project_Type'], 1000), axis=1
)

# ==========================================
# 2. OVERTIME & DURATION ENGINEERING
# ==========================================
# Step 3: Normalize Risk & Resources (0 to 1 scale)
df['Risk_Factor'] = df['Risk_Assessment_Score'] / 100.0
df['Res_Constraint'] = 1 - (df['Resource_Allocation_Score'] / 100.0)

# Step 4: Calculate Component Delays
# Risk component (scaled by complexity)
df['Risk_OT'] = df['Time_Estimate_Days'] * df['Risk_Factor'] * (df['Scope_Complexity_Numeric'] / 3.0)
# Resource bottleneck component
df['Res_OT'] = df['Time_Estimate_Days'] * df['Res_Constraint']
# Historical trend component (using cost deviation as proxy)
df['Hist_OT'] = df['Time_Estimate_Days'] * (df['Historical_Cost_Deviation_%'] / 100.0)

# Step 5: Synthesize Final Duration
df['Estimated_Overtime_Days'] = (df['Risk_OT'] + df['Res_OT'] + df['Hist_OT']) / 3
df['Total_Projected_Duration'] = df['Time_Estimate_Days'] + df['Estimated_Overtime_Days']

# ==========================================
# 3. MACHINE LEARNING PIPELINE
# ==========================================
# Step 6: Feature Selection
# X includes inputs that drive duration; y includes the new Total Duration target
X = df[['Engineered_Area', 'Scope_Complexity_Numeric', 'Resource_Allocation_Score']]
y = df[['Historical_Cost_Deviation_%', 'Time_Estimate_Days', 'Risk_Assessment_Score', 'Total_Projected_Duration']]

# Step 7: Scaling
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_x.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# Step 8: Model Training (MLP Emulator)
model = MLPRegressor(hidden_layer_sizes=(32, 32), max_iter=2000, random_state=42)
model.fit(X_scaled, y_scaled)

# Step 9: Truth Mapping (For Validation)
label_map = {'Feasible': 1.0, 'Borderline': 0.5, 'Not Feasible': 0.0}
df['Ground_Truth_PHC'] = df['Feasibility_Label'].map(label_map)

# ==========================================
# 4. DEPLOYMENT PREPARATION
# ==========================================
# Step 10: Serialization
joblib.dump(model, 'phi_model.pkl')
joblib.dump(scaler_x, 'scaler_x.pkl')
joblib.dump(scaler_y, 'scaler_y.pkl')

print("Process Complete!")
print(f"Model saved with Total Duration capability.")
print(f"Sample Calculation: Baseline {df['Time_Estimate_Days'].iloc[0]} days -> Projected {df['Total_Projected_Duration'].iloc[0]:.1f} days")
