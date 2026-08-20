# -*- coding: utf-8 -*-
"""

Created on Wed Nov  5 16:02:26 2025

@author: Sleepyheadzzzzzz


========================================
Script 5: VIF Analysis
Checks Variance Inflation Factor to identify multicollinearity.
========================================


PURPOSE:
- Compares Multicollinearity (VIF).
- TARGET: VIF values should ideally be < 5.
"""



import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler

# =============================================================================
# --- CONFIGURATION -----------------------------------------------------------
# =============================================================================

INPUT_FILE = "F:/mobile/tree/cluster_data.csv" # Output from Script 4

FEATURES_TO_CHECK = [
    'Initial_TD',
    'Initial_H',          
    'lightemiss',
    'avg_noise_day',
    'avg_noise_night',    
    'Density_10',
    'Density_15',
    'Density25',       
    'Mono_Rate',       
    'avg_svf',
    'avg_radiation',
    'avg_LST',            
    'soil_has_clay',
    'soil_has_infill',
    'soil_has_silt_sand',
    'soil_has_moraine',
    'soil_has_bedrock'
]

# =============================================================================
# --- MAIN SCRIPT -------------------------------------------------------------
# =============================================================================

def calculate_vif(df, features):
    df_clean = df[features].dropna()
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(df_clean), columns=df_clean.columns)
    
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X_scaled.columns
    vif_data["VIF"] = [variance_inflation_factor(X_scaled.values, i) for i in range(X_scaled.shape[1])]
    
    return vif_data.sort_values(by="VIF", ascending=False)

def main():
    print("Loading Data...")
    try:
        df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print("File not found.")
        return

    # Pre-processing to ensure columns exist (Mocking period data if needed)
    # Assuming 'Initial_H' exists or is derived from H15/H17
    if 'Initial_H' not in df.columns and 'H15' in df.columns:
        df['Initial_H'] = df['H15']

    print("\n--- VIF RESULTS ---")
    results = calculate_vif(df, FEATURES_TO_CHECK)
    print(results)

if __name__ == "__main__":
    main()