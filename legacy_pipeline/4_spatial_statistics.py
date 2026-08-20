# -*- coding: utf-8 -*-
"""

Created on Wed Nov  5 16:02:26 2025

@author: Sleepyheadzzzzzz


========================================
Script 4: Spatial Statistics
DBSCAN Clustering, Regime Classification, and Vulnerability Analysis.
========================================

PURPOSE:
1. IDENTIFY CLUSTERS: Uses DBSCAN to find physical groups of trees (50m radius).
2. CLASSIFY REGIMES: 
   - Monoculture: Dominant species ratio > 0.7. (EXCLUDING General categories 1 & 2 -> Marked 'Unknown')
   - Intermediate: Dominant species ratio between 0.3 and 0.7.
   - Mixed: Dominant species ratio < 0.3.
   - Sparse: Isolated trees (DBSCAN Noise).
3. STATISTICS: Compare Carbon Growth between Vulnerable vs Non-Vulnerable areas.
4. VISUALIZATION: 
   - Cluster Map with SMOOTHED outlines.
   - Vulnerability Map colored by stress intensity (Yellow -> Red).
5. NEW METRICS:
   - Density25: Number of neighbors within 25m.
   - Mono_Rate: Ratio of same-species neighbors (with special handling for General species).
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.cluster import DBSCAN
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial import ConvexHull, cKDTree
from scipy.interpolate import splprep, splev

# =============================================================================
# --- CONFIGURATION -----------------------------------------------------------
# =============================================================================

INPUT_FILE = "F:/mobile/tree/tree_all4fi.csv"
OUTPUT_FOLDER = "F:/mobile/analysis/ana60_CLUSTER_VULNERABILITY/"

# DBSCAN
EPS_DISTANCE = 25       # Meters
MIN_SAMPLES = 10        # Minimum trees per cluster

# REGIMES
THRESHOLD_MONO_MIN = 0.70
THRESHOLD_MIXED_MAX = 0.30

# METRICS
DENSITY_RADIUS = 25.0

SPECIES_MAP = {
    1: 'General_Conifer', 2: 'General_Broadleaf', 3: 'Acer',
    4: 'Alnus', 5: 'Betula', 6: 'Pinus', 7: 'Prunus',
    8: 'Quercus', 9: 'Sorbus', 10: 'Tilia', 11: 'Ulmus'
}
GENERAL_SPECIES = ['General_Conifer', 'General_Broadleaf']

# =============================================================================
# --- MAIN SCRIPT -------------------------------------------------------------
# =============================================================================

def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    print(f"Loading {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)
    df = df[df['type'].isin(['Katu', 'Puisto'])].copy()
    df['Species_Name'] = df['Species'].map(SPECIES_MAP)

    # 1. Clustering
    print("Running DBSCAN...")
    coords = df[['X', 'Y']].values
    db = DBSCAN(eps=EPS_DISTANCE, min_samples=MIN_SAMPLES).fit(coords)
    df['Cluster_ID'] = db.labels_
    
    # 2. Regime Classification
    print("Classifying Regimes...")
    regime_map = {}
    for cid, data in df[df['Cluster_ID'] != -1].groupby('Cluster_ID'):
        counts = data['Species_Name'].value_counts(normalize=True)
        dom_spec = counts.index[0]
        dom_ratio = counts.iloc[0]
        
        if dom_ratio > THRESHOLD_MONO_MIN:
            regime_map[cid] = 'Unknown' if dom_spec in GENERAL_SPECIES else 'Monoculture'
        elif dom_ratio < THRESHOLD_MIXED_MAX:
            regime_map[cid] = 'Mixed'
        else:
            regime_map[cid] = 'Intermediate'
            
    df['Regime'] = df['Cluster_ID'].map(regime_map).fillna('Sparse')

    # 3. Density & Mono Rate
    print("Calculating Density25...")
    tree = cKDTree(coords)
    indices = tree.query_ball_point(coords, r=DENSITY_RADIUS)
    
    densities, mono_rates = [], []
    valid_mono_rates = []
    sp_vals = df['Species_Name'].values
    
    for i, neighbors in enumerate(indices):
        n_count = len(neighbors) - 1
        densities.append(n_count)
        
        if n_count > 0:
            others = [x for x in neighbors if x != i]
            same = np.sum(sp_vals[others] == sp_vals[i])
            rate = same / n_count
            mono_rates.append(rate)
            if sp_vals[i] not in GENERAL_SPECIES:
                valid_mono_rates.append(rate)
        else:
            mono_rates.append(0.0)
            
    avg_mono = np.mean(valid_mono_rates) if valid_mono_rates else 0
    df['Density25'] = densities
    df['Mono_Rate'] = [avg_mono if s in GENERAL_SPECIES else r for s, r in zip(sp_vals, mono_rates)]
    
    # 4. Save
    output_csv = os.path.join(OUTPUT_FOLDER, "cluster_analysis_data.csv")
    df.to_csv(output_csv, index=False)
    print(f"Saved to {output_csv}")

if __name__ == "__main__":
    main()