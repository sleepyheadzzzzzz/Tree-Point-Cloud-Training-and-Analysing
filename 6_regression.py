# -*- coding: utf-8 -*-
"""

Created on Wed Nov  5 16:02:26 2025

@author: Sleepyheadzzzzzz


========================================
Script 6: Regression Analysis & Tile Matrix
Trains Linear (Log-Log) and Non-Linear (RF) models, outputs performance metrics,
and generates a Tile Matrix plot of coefficients.
========================================

PURPOSE:
1. Run Log-Log Regression (Linear) per species AND for "General Tree" (All).
2. Train Non-Linear Regression (Random Forest).
3. Save all models (Linear & Non-Linear) and Performance Metrics.
4. Generate "Tile Matrix" Plot with SPECIFIC SORT ORDER.
5. UPDATED: Uses Test-Set R2 for performance metrics to prevent overfitting.
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import seaborn as sns
import os
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# =============================================================================
# --- USER CONFIGURATION ------------------------------------------------------
# =============================================================================

# IO PATHS
INPUT_FILE = "F:/mobile/tree/tree_carbon_updated.csv"
OUTPUT_FOLDER = "F:/mobile/analysis/ana83/"
MODEL_OUTPUT_FOLDER_NAME = "models"

# SPECIES MAPPING
SPECIES_MAP = {
    1: 'General_Conifer', 2: 'General_Broadleaf', 3: 'Acer',
    4: 'Alnus', 5: 'Betula', 6: 'Pinus', 7: 'Prunus',
    8: 'Quercus', 9: 'Sorbus', 10: 'Tilia', 11: 'Ulmus'
}

# FEATURES & TARGETS
CATEGORICAL_FEATURES_BASE = ['type'] 
POTENTIAL_FEATURES = [
    'Log_Initial_H', 'avg_noise_day', 'Density25',        
    'Mono_Rate', 'avg_svf', 'avg_radiation', 'avg_LST',          
    'lightemiss', 'soil_has_infill', 'soil_has_moraine', 'soil_has_bedrock'
]

TARGET_VARIABLE_RAW = 'Annual_Carbon_Growth'
TARGET_VARIABLE_LOG = 'Log_Annual_Carbon_Growth'

# FILTERING THRESHOLDS
MIN_OBS_FOR_VAR = 200        # Minimum observations to include a variable
FINAL_SAMPLE_THRESHOLD = 500 # Minimum samples to analyze a species

# VISUALIZATION
TILE_MATRIX_SORT_PRIORITY = [
    'General Tree', 'General_Broadleaf', 'General_Conifer', 
    'Acer', 'Alnus', 'Betula'
]

# =============================================================================
# --- MAIN SCRIPT -------------------------------------------------------------
# =============================================================================

# Global list to store R2 and RMSE
PERFORMANCE_METRICS = []

def create_long_dataframe(df):
    print("Pre-processing data...")
    # Filter Type
    df = df[df['type'].isin(['Katu', 'Puisto'])].copy()
    
    # Period mappings
    periods = {
        '15_17': {'start_yr': '15', 'end_yr': '17', 'rad': 'ra15_17', 'lst': 'LST_1516'},
        '17_21': {'start_yr': '17', 'end_yr': '21', 'rad': 'ra17_21', 'lst': 'LST_1720'},
        '21_23': {'start_yr': '21', 'end_yr': '23', 'rad': 'ra21_23', 'lst': 'LST_2122'}
    }
    
    # Clean columns
    cols_to_clean = ['noise17d', 'noise22d', 'lightemiss', 
                     'svf15', 'svf17', 'svf21', 'svf23',
                     'ra21_23', 'ra17_21', 'ra15_17',
                     'LST_1516', 'LST_1720', 'LST_2122',
                     'H15', 'H17', 'H21', 'H23',
                     'Density25', 'Mono_Rate']
    
    for col in cols_to_clean:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            if col not in ['Density25', 'Mono_Rate']:
                df[col] = df[col].replace(0, np.nan)
            
    df['avg_noise_day'] = df[['noise17d', 'noise22d']].mean(axis=1)
    df['Species_Name'] = df['Species'].map(SPECIES_MAP)
    
    # Soil dummy creation
    soil_str = df['soil'].fillna('').str.lower()
    df['soil_has_infill'] = soil_str.str.contains('t').astype(int)
    df['soil_has_silt_sand'] = soil_str.str.contains('s\+h').astype(int) 
    df['soil_has_clay'] = soil_str.str.contains('sa').astype(int)        
    df['soil_has_moraine'] = soil_str.str.contains('mr').astype(int)
    df['soil_has_bedrock'] = soil_str.str.contains('ka').astype(int)
    
    all_periods_data = []
    for p_name, p_info in periods.items():
        start_yr = p_info['start_yr']
        period_df = df.copy()
        
        if f'Ann_{p_name}' not in period_df.columns: continue

        period_df[TARGET_VARIABLE_RAW] = period_df[f'Ann_{p_name}']
        period_df['Initial_H'] = period_df[f'H{start_yr}']
        period_df['avg_svf'] = df[[f'svf{start_yr}', f'svf{p_info["end_yr"]}']].mean(axis=1)
        period_df['avg_radiation'] = df[p_info['rad']]
        period_df['avg_LST'] = df[p_info['lst']]
        period_df['Period'] = p_name
        
        period_df = period_df.dropna(subset=[TARGET_VARIABLE_RAW, 'Initial_H'])
        period_df = period_df[period_df['Initial_H'] > 0]
        period_df = period_df[period_df[TARGET_VARIABLE_RAW] > 0]
        
        period_df[TARGET_VARIABLE_LOG] = np.log1p(period_df[TARGET_VARIABLE_RAW])
        period_df['Log_Initial_H'] = np.log1p(period_df['Initial_H'])
        
        cols_to_keep = [TARGET_VARIABLE_LOG, 'Period', 'Species_Name', 'X', 'Y'] + CATEGORICAL_FEATURES_BASE + POTENTIAL_FEATURES
        valid_cols = [c for c in cols_to_keep if c in period_df.columns]
        period_df = period_df[valid_cols]
        all_periods_data.append(period_df)

    if not all_periods_data:
        return pd.DataFrame()
        
    long_df = pd.concat(all_periods_data, ignore_index=True)

    for col in POTENTIAL_FEATURES:
        if col in long_df.columns:
            if col == 'Log_Initial_H': continue
            median_val = long_df[col].median()
            if pd.isna(median_val): median_val = 0 
            long_df[col] = long_df[col].fillna(median_val)
            
    return long_df

def get_significance_label(p):
    if p < 0.001: return "* * *"
    if p < 0.01: return "* *"
    if p < 0.05: return "*"
    return "ns"

def analyze_species(species_name, species_data, model_output_folder):
    current_n = len(species_data)
    if current_n <= FINAL_SAMPLE_THRESHOLD:
        print(f"  Skipping {species_name}: Final N ({current_n}) <= Threshold ({FINAL_SAMPLE_THRESHOLD})")
        return None

    selected_features = []
    
    always_keep = [
        'Log_Initial_H', 'avg_noise_day', 'Density25',
        'avg_svf', 'avg_radiation', 'avg_LST', 'lightemiss'
    ]
    selected_features.extend(always_keep)
    
    # Conditional Mono_Rate
    if species_name not in ['General_Conifer', 'General_Broadleaf']:
        selected_features.append('Mono_Rate')
    
    binary_vars = ['soil_has_infill', 'soil_has_moraine', 'soil_has_bedrock']
    for var in binary_vars:
        if var in species_data.columns:
            if species_data[var].sum() >= MIN_OBS_FOR_VAR:
                selected_features.append(var)

    X = pd.get_dummies(species_data[selected_features + CATEGORICAL_FEATURES_BASE], 
                       columns=CATEGORICAL_FEATURES_BASE, drop_first=True)
    y = species_data[TARGET_VARIABLE_LOG]
    X = X.loc[:, X.var() > 0]
    
    if X.empty: return None

    scaler = StandardScaler()
    cols_to_scale = [c for c in selected_features if c in X.columns]
    if cols_to_scale:
        X[cols_to_scale] = scaler.fit_transform(X[cols_to_scale])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    try:
        # 1. Linear Model (OLS)
        # Full model for inference
        X_sm_full = sm.add_constant(X)
        linear_model_full = sm.OLS(y, X_sm_full).fit()
        joblib.dump(linear_model_full, os.path.join(model_output_folder, f"linear_{species_name}.pkl"))
        r2_lin_full = linear_model_full.rsquared

        # Split model for metrics
        lin_reg_test = LinearRegression()
        lin_reg_test.fit(X_train, y_train)
        y_pred_lin_test = lin_reg_test.predict(X_test)
        rmse_lin_test = np.sqrt(mean_squared_error(y_test, y_pred_lin_test))
        r2_lin_test = r2_score(y_test, y_pred_lin_test)

        # 2. Non-Linear Model (RF)
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf_model.fit(X_train, y_train)
        joblib.dump(rf_model, os.path.join(model_output_folder, f"nonlinear_{species_name}.pkl"))
        y_pred_rf_test = rf_model.predict(X_test)
        rmse_rf_test = np.sqrt(mean_squared_error(y_test, y_pred_rf_test))
        r2_rf_test = r2_score(y_test, y_pred_rf_test)

        # Metrics
        PERFORMANCE_METRICS.append({
            'Species': species_name, 'Model_Type': 'Linear (OLS)',
            'R2': r2_lin_test, 'RMSE': rmse_lin_test, 'N_Samples': current_n, 'Note': 'Test Set'
        })
        PERFORMANCE_METRICS.append({
            'Species': species_name, 'Model_Type': 'Non-Linear (RF)',
            'R2': r2_rf_test, 'RMSE': rmse_rf_test, 'N_Samples': current_n, 'Note': 'Test Set'
        })

        results = []
        for var in linear_model_full.params.index:
            if var == 'const': continue
            results.append({
                'Species': species_name,
                'Variable': var,
                'Coef': linear_model_full.params[var],
                'P_Value': linear_model_full.pvalues[var],
                'Label': get_significance_label(linear_model_full.pvalues[var]),
                'R2': r2_lin_full,
                'N': current_n
            })
        return results

    except Exception as e:
        print(f"Error analyzing {species_name}: {e}")
        return None

def plot_tile_matrix(df_results):
    print("Generating Tile Matrix with Custom Sort...")
    unique_in_data = df_results['Species'].unique()
    
    # Sort Logic
    sorted_species = [s for s in TILE_MATRIX_SORT_PRIORITY if s in unique_in_data]
    others = sorted([s for s in unique_in_data if s not in TILE_MATRIX_SORT_PRIORITY])
    species_list = sorted_species + others

    var_list = df_results['Variable'].unique()
    
    species_labels = []
    for s in species_list:
        subset = df_results[df_results['Species'] == s]
        if not subset.empty:
            r2_val = subset['R2'].iloc[0]
            n_val = subset['N'].iloc[0]
            species_labels.append(f"{s}\n(N={n_val}, R²={r2_val:.2f})")
        else:
            species_labels.append(s)

    fig, ax = plt.subplots(figsize=(len(var_list) * 1.3, len(species_list) * 1.1))
    cmap = plt.get_cmap('RdBu_r') 
    norm = mcolors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1) 

    for i, species in enumerate(species_list):
        for j, variable in enumerate(var_list):
            subset = df_results[(df_results['Species'] == species) & (df_results['Variable'] == variable)]
            
            if not subset.empty:
                coef = subset['Coef'].values[0]
                label = subset['Label'].values[0]
                color = cmap(norm(coef))
                
                rect = mpatches.Rectangle((j - 0.5, i - 0.5), 1, 1, 
                                          facecolor=color, edgecolor='none')
                ax.add_patch(rect)
                
                text_color = 'white' if abs(coef) > 0.5 else 'black'
                
                ax.text(j, i + 0.05, f"{coef:.2f}", 
                        ha='center', va='center', color=text_color, fontsize=11)
                ax.text(j, i - 0.18, label, 
                        ha='center', va='center', color=text_color, fontsize=9)

    ax.set_xticks(range(len(var_list)))
    ax.set_xticklabels(var_list, rotation=45, ha='right', fontsize=10)
    ax.set_yticks(range(len(species_list)))
    ax.set_yticklabels(species_labels, fontsize=10)
    
    ax.set_xlim(-0.5, len(var_list) - 0.5)
    ax.set_ylim(len(species_list) - 0.5, -0.5) 
    ax.set_title("Analysis Results: Coefficient Strength & Significance", fontsize=14, pad=20)
    
    for spine in ax.spines.values(): spine.set_visible(False)
    ax.grid(False)

    sm_scalar = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm_scalar.set_array([])
    cbar = plt.colorbar(sm_scalar, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Coefficient Value', rotation=270, labelpad=15)
    cbar.outline.set_visible(False)

    legend_elements = [
        mpatches.Patch(color='none', label='Significance Levels:'),
        mpatches.Patch(color='none', label='* * * : p < 0.001'),
        mpatches.Patch(color='none', label='* * : p < 0.01'),
        mpatches.Patch(color='none', label='* : p < 0.05'),
        mpatches.Patch(color='none', label='ns    : p >= 0.05')
    ]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.05, 0.0), frameon=False, fontsize=9)

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_FOLDER, "TILE_MATRIX_REFINED.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight') 
    print(f"Plot saved to: {output_path}")

def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    model_folder = os.path.join(OUTPUT_FOLDER, MODEL_OUTPUT_FOLDER_NAME)
    os.makedirs(model_folder, exist_ok=True)

    print(f"Loading data from {INPUT_FILE}...")
    try:
        df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print(f"Error: File not found at {INPUT_FILE}")
        return
        
    data_full = create_long_dataframe(df)
    if data_full.empty:
        print("Data processing returned empty dataframe. Check input columns.")
        return

    unique_species = data_full['Species_Name'].dropna().unique()
    all_results_list = []
    
    # 1. Analyze Individual Species
    for species in unique_species:
        subset = data_full[data_full['Species_Name'] == species].copy()
        if len(subset) > 100:
            print(f"Analyzing {species}...")
            res = analyze_species(species, subset, model_folder)
            if res:
                all_results_list.extend(res)

    # 2. Analyze "General Tree" (Combined)
    print("Analyzing General Tree (Combined)...")
    res_overall = analyze_species("General Tree", data_full, model_folder)
    if res_overall:
        all_results_list.extend(res_overall)

    # 3. Save Performance Metrics
    if PERFORMANCE_METRICS:
        perf_df = pd.DataFrame(PERFORMANCE_METRICS)
        perf_path = os.path.join(OUTPUT_FOLDER, "model_performance_metrics.csv")
        perf_df.to_csv(perf_path, index=False)
        print(f"Performance metrics saved to: {perf_path}")

    # 4. Generate Plot
    if all_results_list:
        full_results_df = pd.DataFrame(all_results_list)
        
        # Clean variable names for plotting
        full_results_df['Variable'] = full_results_df['Variable'].str.replace('type_', '')
        full_results_df['Variable'] = full_results_df['Variable'].str.replace('soil_has_', '')
        full_results_df['Variable'] = full_results_df['Variable'].str.replace('Log_Initial_H', 'Height (Log)')
        full_results_df['Variable'] = full_results_df['Variable'].str.replace('Density25', 'Density (25m)')
        full_results_df['Variable'] = full_results_df['Variable'].str.replace('Mono_Rate', 'Biotic Rate')
        
        plot_tile_matrix(full_results_df)
        
        full_results_df.to_csv(os.path.join(OUTPUT_FOLDER, "matrix_data_raw.csv"), index=False)
    else:
        print("No results generated.")

if __name__ == "__main__":
    main()