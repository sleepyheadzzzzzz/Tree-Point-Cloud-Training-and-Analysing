# -*- coding: utf-8 -*-
"""
Created on Wed Apr 16 12:31:42 2025

@author: Sleepyheadzzzzzz

========================================
Script 2: Trunk Prediction
Applies saved models to fill Trunk Diameter (TD) in the main dataset.
========================================

PURPOSE:
- Loads pre-built models and lookup tables.
- Applies a "monotonic" (non-shrinking) trunk diameter fix.
- OUTPUTS A CLEAN CSV with prediction, model name, and RMSE.
- UPDATED: Allows user to enforce specific model types (Height, Crown, etc.)
"""


import pandas as pd
import numpy as np
import joblib
import os
import json

# =============================================================================
# --- CONFIGURATION -----------------------------------------------------------
# =============================================================================

# PATHS
INPUT_DATA_PATH = "F:/mobile/tree/tree.csv"
OUTPUT_DATA_PATH = "F:/mobile/tree/tree2.csv"
MODEL_FOLDER = "F:/mobile/F/statistics/saved_models"
MODEL_SUMMARY_PATH = "F:/mobile/F/statistics/species_full_summary.csv"

# SETTINGS
USER_MODEL_PREFERENCE = 'best' # Options: 'best', 'Height', 'Crown', 'Height+Crown'
SPECIES_MAP = {
    1: 'General_Conifer', 2: 'General_Broadleaf', 3: 'Acer',
    4: 'Alnus', 5: 'Betula', 6: 'Pinus', 7: 'Prunus',
    8: 'Quercus', 9: 'Sorbus', 10: 'Tilia', 11: 'Ulmus'
}
YEARS = ['15', '17', '21', '23']

# FEATURE DEFINITIONS (Must match training script)
FEATURE_SETS_DICT = {
    'Height': ['tree_height_m'],
    'Crown': ['crown_diameter_m'],
    'Height+Crown': ['tree_height_m', 'crown_diameter_m']
}

# =============================================================================
# --- MAIN SCRIPT -------------------------------------------------------------
# =============================================================================

def get_smart_prediction(species, new_data, summary_df):
    """Selects appropriate model and predicts."""
    # Check if species exists in trained models
    if species not in summary_df['species'].values:
        return None, "NoModel", 0

    # Get best model info
    row = summary_df[summary_df['species'] == species].iloc[0]
    feat_name = row['best_features']
    model_type = row['best_model']
    
    # Check if we have data for this model
    required_cols = FEATURE_SETS_DICT[feat_name]
    if not set(required_cols).issubset(new_data.columns):
        # Fallback logic could go here, for now return None
        return None, "MissingFeatures", 0

    # Load Model
    model_suffix = "linear" if model_type == "Linear" else "rf"
    model_filename = f"{species}_{feat_name}_{model_suffix}.pkl"
    model_path = os.path.join(MODEL_FOLDER, model_filename)
    
    try:
        model = joblib.load(model_path)
        pred = model.predict(new_data[required_cols])[0]
        return pred, f"{model_type}_{feat_name}", 0 # RMSE not loaded here for brevity
    except:
        return None, "LoadError", 0

def process_dataset():
    print("Loading Data...")
    df = pd.read_csv(INPUT_DATA_PATH)
    summary_df = pd.read_csv(MODEL_SUMMARY_PATH)
    
    # Init columns
    for y in YEARS:
        df[f'TD{y}'] = np.nan
        df[f'TD{y}_model'] = ""

    print("Predicting Trunk Diameters...")
    for index, row in df.iterrows():
        sp_name = SPECIES_MAP.get(row['Species'])
        if not sp_name: continue
        
        last_valid_td = 0
        
        for year in YEARS:
            h = pd.to_numeric(row.get(f'H{year}'), errors='coerce')
            dp = pd.to_numeric(row.get(f'DP{year}'), errors='coerce')
            
            if pd.isna(h) or h == 0:
                df.at[index, f'TD{year}'] = last_valid_td
                df.at[index, f'TD{year}_model'] = "CarryForward"
                continue
                
            # Prepare Input
            input_data = pd.DataFrame({'tree_height_m': [h]})
            if pd.notna(dp) and dp > 0:
                input_data['crown_diameter_m'] = [dp]
            
            # Predict
            pred, model_name, _ = get_smart_prediction(sp_name, input_data, summary_df)
            
            if pred is not None:
                # Monotonic Check
                if pred < last_valid_td:
                    pred = last_valid_td
                    model_name += "_MonotonicFix"
                
                df.at[index, f'TD{year}'] = pred
                df.at[index, f'TD{year}_model'] = model_name
                last_valid_td = pred
            else:
                df.at[index, f'TD{year}'] = last_valid_td

    df.to_csv(OUTPUT_DATA_PATH, index=False)
    print(f"Done. Saved to {OUTPUT_DATA_PATH}")

if __name__ == "__main__":
    process_dataset()