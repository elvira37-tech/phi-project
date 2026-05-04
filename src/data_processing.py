import pandas as pd
import numpy as np
import random
import os

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

    if is_average:
        actual_target_sum = target_total * len(STAGES)
    else:
        actual_target_sum = target_total

    ratio = actual_target_sum / (current_sum + 1e-6)
    normalized = [v * ratio for v in raw_vals]
    return normalized

def process_raw_data(input_path, output_path):
    df_raw = pd.read_csv(input_path)
    processed_list = []

    print(f"Starting processing for {len(df_raw)} samples...")

    for _, original_row in df_raw.iterrows():
        row = original_row.to_dict()
        p_type = str(row['Project_Type'])

        base_cost = float(row['Estimated_Cost_USD'])
        base_time = float(row['Time_Estimate_Days'])
        base_risk = float(row['Risk_Assessment_Score'])
        complexity = float(row['Scope_Complexity_Numeric'])
        hist_dev = float(row['Historical_Cost_Deviation_%']) / 100
        resources = float(row.get('Resources', 5))

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

    df_calculated = pd.DataFrame(processed_list)
    df_calculated.to_csv(output_path, index=False)
    print(f"Processing Complete: {len(df_calculated)} rows saved to '{output_path}'.")
    return df_calculated

def generate_synthetic_data(input_path, output_path, target_rows=21000):
    df = pd.read_csv(input_path)
    project_col = 'Project_Type'
    discrete_cols = ['Scope_Complexity_Numeric']
    percentage_cols = ['Risk_Assessment_Score', 'Resource_Allocation_Score']
    linear_cols = ['Engineered_Area', 'Actual_Cost_Total', 'Actual_Time_Total', 'Time_Estimate_Days']

    weights = df[project_col].value_counts(normalize=True)
    needed_count = target_rows - len(df)
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

    df_final.to_csv(output_path, index=False)
    print(f"Synthetic Data generation complete: {len(df_final)} samples saved to '{output_path}'")
    return df_final
