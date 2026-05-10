import pandas as pd
import numpy as np
import random
import jax
import jax.numpy as jnp
from flax import nnx
import optax
import joblib
import time

# --- 1. DATA GENERATION & PREPROCESSING ---
STAGES = ['Site_Prep', 'Foundations', 'Structure', 'Systems', 'Finishing']
UNIT_COST_RANGES = {
    'Building': (2000, 3200),
    'Bridge': (4500, 6500),
    'Power Plant': (6000, 9000),
    'Water Infra': (1800, 2600),
    'Road': (300, 700)
}
RANGES = {
    'Building': {'Time': [(5,10),(10,15),(15,25),(20,30),(25,35)], 'Cost': [(3,7),(10,15),(20,30),(20,35),(15,25)], 'Risk': [(40,50),(80,90),(30,45),(50,65),(20,35)]},
    'Bridge': {'Time': [(5,10),(20,30),(30,40),(10,15),(10,15)], 'Cost': [(5,8),(15,25),(35,45),(10,15),(10,15)], 'Risk': [(35,45),(85,95),(75,85),(30,40),(15,25)]},
    'Power Plant': {'Time': [(10,15),(15,20),(15,20),(30,45),(10,20)], 'Cost': [(5,10),(10,15),(15,20),(45,60),(10,15)], 'Risk': [(50,60),(70,80),(40,50),(85,95),(70,85)]},
    'Water Infra': {'Time': [(15,20),(20,25),(25,35),(15,20),(10,15)], 'Cost': [(10,15),(15,20),(35,45),(15,25),(5,10)], 'Risk': [(75,85),(90,95),(35,50),(50,65),(60,75)]},
    'Road': {'Time': [(30,40),(15,20),(20,25),(10,15),(10,15)], 'Cost': [(20,30),(15,20),(25,30),(10,15),(20,25)], 'Risk': [(85,95),(60,75),(40,50),(25,35),(20,35)]}
}

def get_normalized_values(asset_type, metric, target_total, is_average=False):
    ranges = RANGES[asset_type][metric]
    raw_vals = [random.uniform(low, high) for (low, high) in ranges]
    current_sum = sum(raw_vals)
    actual_target_sum = target_total * len(STAGES) if is_average else target_total
    ratio = actual_target_sum / current_sum
    return [v * ratio for v in raw_vals]

print("Loading raw data...")
df_raw = pd.read_csv('Engineering_Cost_Feasibility_Dataset.csv')
processed_list = []
print(f"Processing {len(df_raw)} samples...")

for _, original_row in df_raw.iterrows():
    row = original_row.to_dict()
    p_type = str(row['Project_Type'])
    base_cost = float(row['Estimated_Cost_USD'])
    base_time = float(row['Time_Estimate_Days'])
    base_risk = float(row['Risk_Assessment_Score'])
    complexity = float(row['Scope_Complexity_Numeric'])
    hist_dev = float(row['Historical_Cost_Deviation_%']) / 100
    resources = float(row.get('Resource_Allocation_Score', 50)) / 10 # Adjusted based on common notebook patterns

    min_uc, max_uc = UNIT_COST_RANGES[p_type]
    row['Engineered_Area'] = round(base_cost / random.uniform(min_uc, max_uc), 2)

    planned_costs = get_normalized_values(p_type, 'Cost', base_cost)
    planned_days = get_normalized_values(p_type, 'Time', base_time)
    stage_risks = get_normalized_values(p_type, 'Risk', base_risk, is_average=True)

    actual_cost_total = 0
    actual_time_total = 0

    for i, stage in enumerate(STAGES):
        p_cost = planned_costs[i]
        p_days = planned_days[i]
        s_risk = stage_risks[i]

        cost_variance = (1 + hist_dev) * (1 + (complexity * 0.05)) * (1 + (s_risk / 1000))
        res_efficiency = 1 - (resources * 0.01)
        a_cost = round(p_cost * cost_variance * res_efficiency, 2)

        time_variance = (1 + (s_risk / 150)) * (1 + (complexity * 0.08))
        res_delay = 1.3 - (resources / 10)
        a_days = int(round(p_days * time_variance * res_delay))

        row[f'{stage}_Planned_Cost'] = round(p_cost, 2)
        row[f'{stage}_Planned_Days'] = int(round(p_days))
        row[f'{stage}_Risk'] = round(min(s_risk, 100), 2)
        row[f'{stage}_Actual_Cost'] = a_cost
        row[f'{stage}_Actual_Days'] = a_days

        actual_cost_total += a_cost
        actual_time_total += a_days

    row['Actual_Cost_Total'] = round(actual_cost_total, 2)
    row['Actual_Time_Total'] = actual_time_total
    processed_list.append(row)

df_processed = pd.DataFrame(processed_list)
df_processed.to_csv('Processed_Engineering_Data.csv', index=False)
print("Data processing complete.")

# --- 2. SYNTHETIC DATA AUGMENTATION ---
TARGET_ROWS = 21000
df = df_processed
project_col = 'Project_Type'
discrete_cols = ['Scope_Complexity_Numeric']
percentage_cols = ['Risk_Assessment_Score', 'Resource_Allocation_Score']
linear_cols = ['Engineered_Area', 'Actual_Cost_Total', 'Actual_Time_Total', 'Time_Estimate_Days']

weights = df[project_col].value_counts(normalize=True)
needed_count = TARGET_ROWS - len(df)
synthetic_data = []

print(f"Generating {needed_count} synthetic samples...")
for i in range(needed_count):
    p_type = np.random.choice(weights.index, p=weights.values)
    subset = df[df[project_col] == p_type]
    idx = subset.index
    row1, row2 = subset.loc[np.random.choice(idx)], subset.loc[np.random.choice(idx)]
    alpha = np.random.uniform(-0.05, 1.05)
    new_row = {project_col: p_type}
    for col in linear_cols + percentage_cols + discrete_cols:
        val = (row1[col] * alpha) + (row2[col] * (1 - alpha))
        if col in discrete_cols:
            new_row[col] = round(val)
        elif col in percentage_cols:
            noise = np.random.normal(0, 2.0)
            new_row[col] = np.clip(val + noise, 0, 100)
        else:
            noise_level = 0.02 if i % 10 != 0 else 0.05
            noise = np.random.normal(0, noise_level * val)
            new_row[col] = val + noise
    synthetic_data.append(new_row)

df_final = pd.concat([df, pd.DataFrame(synthetic_data)], ignore_index=True)
for col in ['Engineered_Area', 'Actual_Cost_Total']:
    df_final[col] = df_final[col].clip(lower=df[col].min() * 0.8)

df_final.to_csv('Data_21ksamples.csv', index=False)
print("Synthetic data generation complete.")

# --- 3. NEURAL NETWORK TRAINING ---
df = pd.read_csv('Data_21ksamples.csv')
continuous_input_cols = ['Engineered_Area', 'Scope_Complexity_Numeric', 'Risk_Assessment_Score', 'Resource_Allocation_Score', 'Time_Estimate_Days']
cost_stages = [f'{s}_Actual_Cost' for s in STAGES]
time_stages = [f'{s}_Actual_Days' for s in STAGES]
output_cols = cost_stages + time_stages

for col in continuous_input_cols + output_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=continuous_input_cols + output_cols + ['Project_Type'])

project_type_dummies = pd.get_dummies(df['Project_Type'], prefix='Proj')
df = pd.concat([df, project_type_dummies], axis=1)
encoded_project_cols = project_type_dummies.columns.tolist()
param_cols = encoded_project_cols + continuous_input_cols

params = df[param_cols].values.astype(np.float32)
output = df[output_cols].values.astype(np.float32)
num_params, num_output = len(param_cols), len(output_cols)

num_train = int(len(params) * 0.8)
params_train, output_train = params[:num_train], output[:num_train]
params_min, params_max = np.min(params_train, axis=0), np.max(params_train, axis=0)
output_min, output_max = np.min(output_train, axis=0), np.max(output_train, axis=0)

params_range = np.where((params_max - params_min) == 0, 1.0, params_max - params_min)
output_range = np.where((output_max - output_min) == 0, 1.0, output_max - output_min)

X_train = jnp.array((params_train - params_min) / params_range)
Y_train = jnp.array((output_train - output_min) / output_range)

class ConstructionNN(nnx.Module):
    def __init__(self, rngs: nnx.Rngs):
        self.linear1 = nnx.Linear(num_params, 64, rngs=rngs)
        self.linear2 = nnx.Linear(64, 64, rngs=rngs)
        self.linear3 = nnx.Linear(64, num_output, rngs=rngs)
    def __call__(self, x):
        x = nnx.tanh(self.linear1(x))
        x = nnx.tanh(self.linear2(x))
        return self.linear3(x)

model = ConstructionNN(rngs=nnx.Rngs(42))
optimizer = nnx.Optimizer(model, wrt=nnx.Param, tx=optax.adam(learning_rate=1e-4))

def loss_fn(mod, X, Y):
    return jnp.mean((mod(X) - Y)**2)

@nnx.jit
def train_step(mod, opt, X, Y):
    grads = nnx.grad(loss_fn, argnums=0)(mod, X, Y)
    opt.update(mod, grads)

print("Training model...")
for epoch in range(10001):
    train_step(model, optimizer, X_train, Y_train)
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss: {loss_fn(model, X_train, Y_train):.3e}")

combined_assets = {
    'model_state': nnx.state(model, nnx.Param),
    'param_cols': param_cols,
    'output_cols': output_cols,
    'params_min': np.array(params_min),
    'params_max': np.array(params_max),
    'params_range': np.array(params_range),
    'output_min': np.array(output_min),
    'output_max': np.array(output_max),
    'output_range': np.array(output_range)
}
joblib.dump(combined_assets, 'jax_project_model.pkl')
print("Model training and assets saved.")
