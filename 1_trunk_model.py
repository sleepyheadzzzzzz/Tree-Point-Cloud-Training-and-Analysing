# -*- coding: utf-8 -*-
"""
Created on Tue Apr 15 14:31:30 2025

@author: Sleepyheadzzzzzz

========================================
Script 1: Trunk Model Training
Trains prediction models (Linear & RF) for Trunk Diameter using Height and Crown.
========================================


Tree species statistics and modeling workflow
Predict trunk diameter (ell_eq_diameter_m) using:
- Height only, Crown only, Height+Crown
- Linear Regression & Random Forest
Includes:
- Descriptive stats
- Correlation heatmap (NEW)
- Model results (RMSE, R²)
- Linear formulas & RF feature importance
- Multi-panel scatter plots, p-value heatmap, & RF importance plot (NEW)
- Full summary table
- Saved pre-trained LR & RF models (REVISED)
- Helper function to predict trunk diameter for new trees with input checking (REVISED)
"""



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import math
import os
import joblib

# =============================================================================
# --- CONFIGURATION -----------------------------------------------------------
# =============================================================================

# PATHS
INPUT_FILE = "F:/mobile/F/Tree_Results/all_species_summary.csv"
OUTPUT_FOLDER = "F:/mobile/F/statistics"
SAVED_MODELS_SUBFOLDER = "saved_models"

# PARAMETERS
MIN_SAMPLES_PER_SPECIES = 25
RANDOM_SEED = 42
CV_FOLDS = 5
RF_ESTIMATORS = 200

# FEATURES
TARGET_VARIABLE = 'ell_eq_diameter_m'
FEATURE_SETS = {
    'Height': ['tree_height_m'],
    'Crown': ['crown_diameter_m'],
    'Height+Crown': ['tree_height_m', 'crown_diameter_m']
}

# =============================================================================
# --- MAIN SCRIPT -------------------------------------------------------------
# =============================================================================

def main():
    # 0. Setup Directories
    rf_model_folder = os.path.join(OUTPUT_FOLDER, SAVED_MODELS_SUBFOLDER)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    os.makedirs(rf_model_folder, exist_ok=True)

    # 1. Load Data
    print(f"Loading data from {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)

    # 2. Filter Species
    species_counts = df['species'].value_counts()
    valid_species = species_counts[species_counts >= MIN_SAMPLES_PER_SPECIES].index
    df_filtered = df[df['species'].isin(valid_species)]
    print(f"Analyzing {len(valid_species)} species with >= {MIN_SAMPLES_PER_SPECIES} samples each.")

    # 3. Descriptive Stats
    numeric_features = ['tree_height_m', 'crown_diameter_m', TARGET_VARIABLE]
    desc_stats = df_filtered.groupby("species")[numeric_features].describe()
    desc_stats.to_csv(os.path.join(OUTPUT_FOLDER, "species_descriptive_stats.csv"))

    # 3.5 Correlation Matrix
    corr_matrix = df_filtered[numeric_features].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
    plt.title("Feature Correlation Heatmap (All Species)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FOLDER, "feature_correlation_heatmap.png"))
    plt.close()

    # 4. Modeling
    kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    results = []
    coef_pvals = []
    final_formulas = []
    rf_importances = []

    print("\n=== Running models for each species... ===")
    for sp in valid_species:
        df_sp = df_filtered[df_filtered['species'] == sp].copy()
        y = df_sp[TARGET_VARIABLE]

        for feat_name, features in FEATURE_SETS.items():
            X = df_sp[features]

            # --- Linear Regression ---
            lin_pipe = Pipeline([
                ('scaler', StandardScaler()),
                ('regressor', LinearRegression())
            ])
            
            # Cross-Validation Metrics
            y_pred_lin = cross_val_predict(lin_pipe, X, y, cv=kf)
            rmse_lin = mean_squared_error(y, y_pred_lin, squared=False)
            r2_lin = r2_score(y, y_pred_lin)
            
            # Final Fit & Save
            lin_pipe.fit(X, y)
            joblib.dump(lin_pipe, os.path.join(rf_model_folder, f"{sp}_{feat_name}_linear.pkl"))

            # Statsmodels for P-values
            X_const = sm.add_constant(X)
            lin_model_ols = sm.OLS(y, X_const).fit()
            coef = lin_model_ols.params
            pvals = lin_model_ols.pvalues
            
            formula = f"{sp} trunk_diameter = {coef['const']:.3f}"
            for f in features:
                formula += f" + ({coef[f]:.3f} * {f})"
            final_formulas.append({
                "species": sp, "features": feat_name, "model": "Linear", "formula": formula
            })
            for feat, pval in pvals.items():
                if feat != "const":
                    coef_pvals.append({
                        "species": sp, "features": feat_name, "predictor": feat, "p_value": pval
                    })

            # --- Random Forest ---
            rf = RandomForestRegressor(n_estimators=RF_ESTIMATORS, random_state=RANDOM_SEED)
            y_pred_rf = cross_val_predict(rf, X, y, cv=kf)
            rmse_rf = mean_squared_error(y, y_pred_rf, squared=False)
            r2_rf = r2_score(y, y_pred_rf)

            # Final Fit & Save
            rf.fit(X, y)
            joblib.dump(rf, os.path.join(rf_model_folder, f"{sp}_{feat_name}_rf.pkl"))

            rf_importances.append({
                "species": sp, "features": feat_name, "model": "RandomForest",
                "feature_importances": {f: v for f, v in zip(features, rf.feature_importances_)}
            })

            results.append({'species': sp,'features': feat_name,'model': 'Linear','RMSE': rmse_lin,'R2': r2_lin})
            results.append({'species': sp,'features': feat_name,'model': 'RandomForest','RMSE': rmse_rf,'R2': r2_rf})

    # 5. Save Results
    pd.DataFrame(results).to_csv(os.path.join(OUTPUT_FOLDER, "species_modeling_results.csv"), index=False)
    
    df_results = pd.DataFrame(results)
    df_best = df_results.loc[df_results.groupby("species")["R2"].idxmax()].reset_index(drop=True)
    df_best.to_csv(os.path.join(OUTPUT_FOLDER, "species_best_model.csv"), index=False)
    
    pd.DataFrame(coef_pvals).to_csv(os.path.join(OUTPUT_FOLDER, "species_linear_pvalues.csv"), index=False)
    pd.DataFrame(final_formulas).to_csv(os.path.join(OUTPUT_FOLDER, "species_linear_formulas.csv"), index=False)
    
    # 6. Generate Summary Table
    summary_rows = []
    for _, row in df_best.iterrows():
        sp = row['species']
        best_model = row['model']
        best_features = row['features']
        
        # Get formula if Linear
        formula_row = [f['formula'] for f in final_formulas if f['species'] == sp and f['features'] == best_features and f['model'] == 'Linear']
        formula = formula_row[0] if formula_row else "N/A (RF Best)"
        
        summary_rows.append({
            'species': sp, 'best_model': best_model, 'best_features': best_features,
            'best_R2': row['R2'], 'formula_if_linear': formula
        })
    
    pd.DataFrame(summary_rows).to_csv(os.path.join(OUTPUT_FOLDER, "species_full_summary.csv"), index=False)
    print("Modeling complete. Results saved.")

if __name__ == "__main__":
    main()