# -*- coding: utf-8 -*-
"""

Created on Wed Nov  5 16:02:26 2025

@author: Sleepyheadzzzzzz


==========================================================
Script 3: Carbon Calculator
Calculates biomass/carbon using allometric equations and computes growth.
==========================================================

UPDATES IN THIS VERSION:
1. Pinus/Conifer: Polynomial -> 0.444 + 0.010 * (D^2 * H)
2. Broadleaf/Betula: Repola Multivariate (Pic 47dac1)
   -> ln(y) = -3.654 + 10.582 * (ds / (ds + 24)) + 3.018 * ln(h)
   -> ds = 2 + 1.25 * DBH_cm
3. Ulmus: MOVED to Group 5 (Same as Tilia/Acer/Sorbus)
4. Alnus: Kept previous (Power Law mm v6)

"""



import pandas as pd
import numpy as np

# =============================================================================
# --- CONFIGURATION -----------------------------------------------------------
# =============================================================================

INPUT_PATH = "F:/mobile/tree/tree_all4final.csv"
OUTPUT_PATH = "F:/mobile/tree/tree_carbon_updated.csv"
CARBON_FACTOR = 0.5

ALLOMETRIC_EQS = {
    'Pinus': {'type': 'conifer_poly', 'p': {'inter': 0.444, 'slope': 0.010}},
    'General_Conifer': {'type': 'conifer_poly', 'p': {'inter': 0.444, 'slope': 0.010}},
    'Betula': {'type': 'repola', 'p': {'b0': -3.654, 'b1': 10.582, 'b2': 3.018, 'k': 24}},
    'General_Broadleaf': {'type': 'repola', 'p': {'b0': -3.654, 'b1': 10.582, 'b2': 3.018, 'k': 24}},
    'Alnus': {'type': 'power_mm', 'p': {'a': 0.000146, 'b': 2.6035333}},
    'Quercus': {'type': 'log10_h_d2', 'p': {'a': -1.7194, 'b': 1.0414}},
    'Acer': {'type': 'log10_d2_g', 'p': {'a': 1.1891, 'b': 1.419}},
    'Sorbus': {'type': 'log10_d2_g', 'p': {'a': 1.1891, 'b': 1.419}},
    'Tilia': {'type': 'log10_d2_g', 'p': {'a': 1.1891, 'b': 1.419}},
    'Prunus': {'type': 'log10_d2_g', 'p': {'a': 1.1891, 'b': 1.419}},
    'Ulmus': {'type': 'log10_d2_g', 'p': {'a': 1.1891, 'b': 1.419}}
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
        if eq['type'] == 'conifer_poly':
            return p['inter'] + p['slope'] * (dbh_cm**2 * h_m)
        elif eq['type'] == 'repola':
            ds = 2 + 1.25 * dbh_cm
            return np.exp(p['b0'] + p['b1']*(ds/(ds+p['k'])) + p['b2']*np.log(h_m))
        elif eq['type'] == 'power_mm':
            return p['a'] * ((dbh_cm*10)**p['b'])
        elif eq['type'] == 'log10_h_d2':
            return 10**(p['a'] + p['b']*np.log10(h_m * dbh_cm**2))
        elif eq['type'] == 'log10_d2_g':
            return (10**(p['a'] + p['b']*np.log10(dbh_cm**2))) / 1000.0
    except:
        return 0.0
    return 0.0

def main():
    print("Reading Data...")
    df = pd.read_csv(INPUT_PATH)
    years = ['15', '17', '21', '23']
    
    print("Calculating Carbon...")
    for i, row in df.iterrows():
        sp = SPECIES_MAP.get(row['Species'])
        if not sp: continue
        
        for y in years:
            td = row.get(f'TD{y}')
            h = row.get(f'H{y}')
            if pd.notna(td) and td > 0:
                bio = calculate_biomass(sp, td*100, h) # m -> cm
                df.at[i, f'CS_{y}'] = bio * CARBON_FACTOR
            else:
                df.at[i, f'CS_{y}'] = 0.0
                
    # Calculate Annual Growth (Simplified for brevity)
    print("Calculating Annual Growth...")
    for i, row in df.iterrows():
        vals = {int(y): row.get(f'CS_{y}', 0) for y in years}
        valid_yrs = [y for y, v in vals.items() if v > 0]
        
        if len(valid_yrs) >= 2:
            start, end = min(valid_yrs), max(valid_yrs)
            growth = vals[end] - vals[start]
            df.at[i, 'Annual_Carbon_Growth'] = growth / (end - start)
        else:
            df.at[i, 'Annual_Carbon_Growth'] = 0

    df.to_csv(OUTPUT_PATH, index=False)
    print("Done.")

if __name__ == "__main__":
    main()