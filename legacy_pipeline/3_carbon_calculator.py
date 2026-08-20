# -*- coding: utf-8 -*-
"""
Created on Wed Nov  5 16:02:26 2025

@author: Sleepyheadzzzzzz

==========================================================
Script 3: Carbon Calculator
Calculates biomass/carbon using allometric equations and computes growth.
==========================================================

UPDATES IN THIS VERSION:
1. Pinus/Conifer: Updated to Repola Total Biomass Equation
   ln(y) = -3.198 + 9.547 * (ds / (ds + 12)) + 3.241 * (H / (H + 20))
   where ds = 2 + 1.25 * DBH_cm
2. Betula & General Broadleaf: Maintained Repola structure
3. Tilia, Acer, Ulmus, Sorbus, Prunus: Log-transformed equations
4. Alnus & Quercus: Specific power/log models

"""

import pandas as pd
import numpy as np

# =============================================================================
# --- CONFIGURATION -----------------------------------------------------------
# =============================================================================

INPUT_PATH = "F:/mobile/tree/tree_all4fin.csv"
OUTPUT_PATH = "F:/mobile/tree/tree_carbon_updated6.csv"
CARBON_FACTOR = 0.5

ALLOMETRIC_EQS = {
    # --- CONIFERS (Updated from Repola images) ---
    # Coefficients from "Total" column: b0=-3.198, b1=9.547, b2=3.241
    # Denominators: d_ski+12, h+20
    'Pinus':           {'type': 'repola_variable_h', 
                        'p': {'b0': -3.198, 'b1': 9.547, 'k1': 12, 'b2': 3.241, 'k2': 20}},
    'General_Conifer': {'type': 'repola_variable_h', 
                        'p': {'b0': -3.198, 'b1': 9.547, 'k1': 12, 'b2': 3.241, 'k2': 20}},
    
    # --- BETULA & GENERAL BROADLEAF ---
    # ln(y) = b0 + b1 * (ds / (ds + k1)) + b2 * (H / (H + k2))
    'Betula':            {'type': 'repola_variable_h', 
                          'p': {'b0': -3.654, 'b1': 10.582, 'k1': 12, 'b2': 3.018, 'k2': 22}},
    'General_Broadleaf': {'type': 'repola_variable_h', 
                          'p': {'b0': -3.654, 'b1': 10.582, 'k1': 12, 'b2': 3.018, 'k2': 22}},
    
    # --- OTHER BROADLEAVES (Equation 3 from sheet) ---
    # ln(Y) = -1.9958 + 2.3625 * ln(d)
    'Acer':   {'type': 'ln_d_only', 'p': {'inter': -1.9958, 'slope': 2.3625}},
    'Sorbus': {'type': 'ln_d_only', 'p': {'inter': -1.9958, 'slope': 2.3625}},
    'Tilia':  {'type': 'ln_d_only', 'p': {'inter': -1.9958, 'slope': 2.3625}},
    'Prunus': {'type': 'ln_d_only', 'p': {'inter': -1.9958, 'slope': 2.3625}},
    'Ulmus':  {'type': 'ln_d_only', 'p': {'inter': -1.9958, 'slope': 2.3625}},

    # --- ALNUS & QUERCUS ---
    'Alnus':   {'type': 'power_mm', 'p': {'a': 0.000146, 'b': 2.6035333}},
    'Quercus': {'type': 'log10_h_d2', 'p': {'a': -1.7194, 'b': 1.0414}}
}

SPECIES_MAP = {
    1: 'General_Conifer', 2: 'General_Broadleaf', 3: 'Acer',
    4: 'Alnus', 5: 'Betula', 6: 'Pinus', 7: 'Prunus',
    8: 'Quercus', 9: 'Sorbus', 10: 'Tilia', 11: 'Ulmus'
}

# =============================================================================
# --- MAIN SCRIPT -------------------------------------------------------------
# =============================================================================

def calculate_biomass(species, dbh_cm, h_m):
    if species not in ALLOMETRIC_EQS or dbh_cm <= 0: return 0.0
    eq = ALLOMETRIC_EQS[species]
    p = eq['p']
    
    try:
        # 1. Repola Equation (Used for Pinus, Conifer, Betula, Broadleaf)
        # ln(y) = b0 + b1*(ds/(ds+k1)) + b2*(H/(H+k2))
        # ds = 2 + 1.25 * DBH
        if eq['type'] == 'repola_variable_h':
            ds = 2 + 1.25 * dbh_cm
            exponent = p['b0'] + p['b1']*(ds/(ds+p['k1'])) + p['b2']*(h_m/(h_m+p['k2']))
            return np.exp(exponent)
            
        # 2. Log-Transformed (Tilia, Acer, etc.)
        elif eq['type'] == 'ln_d_only':
            # ln(Y) = inter + slope * ln(d)  ->  Y = exp(...)
            return np.exp(p['inter'] + p['slope'] * np.log(dbh_cm))
            
        # 3. Power Law (Alnus)
        elif eq['type'] == 'power_mm':
            return p['a'] * ((dbh_cm*10)**p['b'])
            
        # 4. Log10 H*D2 (Quercus)
        elif eq['type'] == 'log10_h_d2':
            return 10**(p['a'] + p['b']*np.log10(h_m * dbh_cm**2))

        # 5. Legacy/Unused
        elif eq['type'] == 'conifer_poly':
            return p['inter'] + p['slope'] * (dbh_cm**2 * h_m)
            
    except Exception as e:
        return 0.0
    return 0.0

def main():
    print("Reading Data...")
    try:
        df = pd.read_csv(INPUT_PATH)
    except FileNotFoundError:
        print(f"Error: Could not find file at {INPUT_PATH}")
        return

    years = ['15', '17', '21', '23']
    
    print("Calculating Carbon...")
    for i, row in df.iterrows():
        sp = SPECIES_MAP.get(row['Species'])
        if not sp: continue
        
        for y in years:
            td = row.get(f'TD{y}')
            h = row.get(f'H{y}')
            
            # Ensure we have valid numeric data for both TD and H
            if pd.notna(td) and pd.notna(h) and td > 0:
                # TD is assumed to be in meters in the source, converted to cm here
                bio = calculate_biomass(sp, td*100, h) 
                df.at[i, f'CS_{y}'] = bio * CARBON_FACTOR
            else:
                df.at[i, f'CS_{y}'] = 0.0
                
    # Calculate Annual Growth
    print("Calculating Annual Growth...")
    for i, row in df.iterrows():
        # Create a dictionary of {year: carbon_stock} for non-zero years
        vals = {int(y): row.get(f'CS_{y}', 0) for y in years}
        # Filter only years where Carbon Stock > 0
        valid_yrs = [y for y, v in vals.items() if v > 0]
        
        if len(valid_yrs) >= 2:
            start, end = min(valid_yrs), max(valid_yrs)
            growth = vals[end] - vals[start]
            # Avoid division by zero just in case
            time_diff = end - start
            if time_diff > 0:
                df.at[i, 'Annual_Carbon_Growth'] = growth / time_diff
            else:
                df.at[i, 'Annual_Carbon_Growth'] = 0
        else:
            df.at[i, 'Annual_Carbon_Growth'] = 0

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Done. Saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()