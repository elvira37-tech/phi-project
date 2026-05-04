import pandas as pd
import numpy as np
import time
import jax
import jax.numpy as jnp
from flax import nnx
import optax
import joblib
import os

from src.data_processing import process_raw_data, generate_synthetic_data
from src.model import ConstructionNN

def train_model():
    # 1. Data Preparation
    raw_data_path = 'data/Engineering_Cost_Feasibility_Dataset.csv'
    processed_data_path = 'data/Processed_Engineering_Data.csv'
    synthetic_data_path = 'data/Data_21ksamples.csv'

    if not os.path.exists(processed_data_path):
        process_raw_data(raw_data_path, processed_data_path)
    
    if not os.path.exists(synthetic_data_path):
        generate_synthetic_data(processed_data_path, synthetic_data_path)

    df = pd.read_csv(synthetic_data_path)

    # --- CLEANING ---
    project_type_col = 'Project_Type'
    continuous_input_cols = [
        'Engineered_Area', 'Scope_Complexity_Numeric',
        'Risk_Assessment_Score', 'Resource_Allocation_Score', 'Time_Estimate_Days'
    ]
    cost_stages = ['Site_Prep_Actual_Cost', 'Foundations_Actual_Cost', 'Structure_Actual_Cost', 'Systems_Actual_Cost', 'Finishing_Actual_Cost']
    time_stages = ['Site_Prep_Actual_Days', 'Foundations_Actual_Days', 'Structure_Actual_Days', 'Systems_Actual_Days', 'Finishing_Actual_Days']
    output_cols = cost_stages + time_stages

    for col in continuous_input_cols + output_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=continuous_input_cols + output_cols + [project_type_col])

    # --- ONE-HOT ENCODING ---
    project_type_dummies = pd.get_dummies(df[project_type_col], prefix='Proj')
    df = pd.concat([df, project_type_dummies], axis=1)
    encoded_project_cols = project_type_dummies.columns.tolist()

    param_cols = encoded_project_cols + continuous_input_cols

    params = df[param_cols].values.astype(np.float32)
    output = df[output_cols].values.astype(np.float32)

    num_params = len(param_cols)
    num_output = len(output_cols)

    num_total_samples = len(params)
    num_train = int(num_total_samples * 0.8)

    params_train, output_train = params[:num_train], output[:num_train]
    params_valid, output_valid = params[num_train:], output[num_train:]

    # Normalization
    params_min, params_max = np.min(params_train, axis=0), np.max(params_train, axis=0)
    output_min, output_max = np.min(output_train, axis=0), np.max(output_train, axis=0)

    params_range = np.where((params_max - params_min) == 0, 1.0, params_max - params_min)
    output_range = np.where((output_max - output_min) == 0, 1.0, output_max - output_min)

    X_train = jnp.array((params_train - params_min) / params_range)
    Y_train = jnp.array((output_train - output_min) / output_range)

    X_valid = jnp.array((params_valid - params_min) / params_range)
    Y_valid = jnp.array((output_valid - output_min) / output_range)

    # 2. Define the Neural Network
    model = ConstructionNN(num_params, num_output, rngs=nnx.Rngs(42))

    # 3. Define Training Logic
    def loss_fn(mod, X, Y):
        prediction = mod(X)
        return jnp.mean((prediction - Y)**2)

    optimizer = nnx.Optimizer(model, wrt=nnx.Param, tx=optax.adam(learning_rate=1e-4))

    @nnx.jit
    def train_step(mod, opt, X, Y):
        grad_fn = nnx.grad(loss_fn, argnums=0)
        grads = grad_fn(mod, X, Y)
        opt.update(mod, grads)

    # 4. Training Loop
    print(f"Training started on {num_params} inputs to predict {num_output} outputs...")
    for epoch in range(10001):
        train_step(model, optimizer, X_train, Y_train)

        if epoch % 1000 == 0:
            l_train = loss_fn(model, X_train, Y_train)
            l_valid = loss_fn(model, X_valid, Y_valid)
            print(f'Epoch: {epoch:>5} | Train Loss: {l_train:.3e} | Valid Loss: {l_valid:.3e}')

    # 5. Save Model and Assets
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

    file_name = 'jax_project_model.pkl'
    joblib.dump(combined_assets, file_name)
    print(f"\n\u2705 Combined model and assets saved to '{file_name}'")

if __name__ == "__main__":
    train_model()
