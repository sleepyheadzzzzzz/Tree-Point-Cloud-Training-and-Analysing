# Urban Tree Carbon & Spatial Analysis Pipeline

# Urban Tree Carbon & Spatial Analysis Tool

A comprehensive Python workflow for modeling urban tree metrics, estimating carbon sequestration, and analyzing spatial vulnerability. This pipeline integrates machine learning (Random Forest & Linear Regression) with spatial statistics (DBSCAN) to quantify how environmental factors and planting patterns (Monoculture vs. Mixed) impact urban forest growth.

## 🚀 Features

* **Missing Data Imputation:** Uses machine learning to predict missing Trunk Diameters (TD) based on Tree Height and Crown Diameter, enforcing monotonic growth logic (trees cannot shrink over time).
* **Carbon Sequestration:** Calculates biomass and carbon storage using species-specific allometric equations (e.g., Repola, Power Law).
* **Spatial Clustering:** Utilizes DBSCAN to identify physical tree clusters and classify them into planting regimes:
  * **Monoculture:** >70% same species.
  * **Mixed:** <30% same species.
  * **Sparse/Isolated:** Noise points.
* **Vulnerability Analysis:** Computes local density metrics (`Density25`) and biotic competition rates (`Mono_Rate`) to assess environmental stress.
* **Statistical Modeling:** Performs Log-Log Linear Regression and Non-Linear Random Forest analysis to identify significant drivers of annual carbon growth.
* **Visualization:** Generates "Tile Matrix" heatmaps to visualize the strength and significance of environmental coefficients across different species.

## 🔄 Pipeline

The tool consists of 6 sequential scripts. Each output feeds into the next step.

1. **`1_trunk_model.py` (Model Training)**
   * Trains Linear and Random Forest models for every species to predict trunk diameter.
   * Outputs `.pkl` model files and a performance summary.

2. **`2_trunk_predictor.py` (Data Imputation)**
   * Applies the trained models to the main dataset to fill missing trunk diameters.
   * Applies a "monotonic fix" to ensure a tree's diameter does not decrease between years (e.g., from 2015 to 2023).

3. **`3_carbon_calculator.py` (Carbon Estimation)**
   * Converts physical metrics (Height, DBH) into Biomass and Carbon Storage (kg).
   * Calculates the Annual Carbon Growth Rate.

4. **`4_spatial_statistics.py` (Clustering)**
   * Runs DBSCAN clustering on tree coordinates.
   * Calculates `Density25` (neighbors within 25m) and `Mono_Rate` (species homogeneity).

5. **`5_VIF.py` (Multicollinearity Check)**
   * Calculates Variance Inflation Factor (VIF) to ensure regression variables are not highly correlated.

6. **`6_regression.py` (Final Analysis)**
   * Runs the final growth regressions (OLS & RF).
   * Produces the Tile Matrix plot showing which factors (Soil, Noise, Density) significantly affect growth.

## Configurations

* **Input Data:** The scripts require a CSV file structure containing tree IDs, coordinates (X, Y), species, and temporal metric columns (e.g., `H15`, `H17`, `H21` for height in years 2015, 2017, etc.).
* **Key Parameters (Top of Scripts):**
  * **`USER_MODEL_PREFERENCE` (Script 2):** Selects which model logic to use ('best', 'Height', 'Crown').
  * **`ALLOMETRIC_EQS` (Script 3):** Dictionary defining the biomass formulas. Ensure these match your specific region/species.
  * **`EPS_DISTANCE` (Script 4):** The search radius for DBSCAN clustering (Default: 25m).
  * **`THRESHOLD_MONO_MIN` (Script 4):** The percentage threshold to classify a cluster as a "Monoculture" (Default: 0.70 or 70%).

## Remind
* **Sample Data:** The original dataset recording tree traits, environment values, and carbon information is uploaded as a sample named _tree_carbon_updated.csv._
* **Pre-Processing:** There is a pre-step to extract trunk diameter information; please look in another repository: https://github.com/sleepyheadzzzzzz/Tree-point-cloud-trunk-segmentation-and-measurement.
* **Sequential Execution:** You should run the scripts in order (1 → 6). If you skip a step, the subsequent script will miss required columns (e.g., Script 6 requires the `Mono_Rate` column generated in Script 4).
  * But you can use either of the scripts individually if you have your own dataset.
* **Coordinate System:** Ensure your input CSV uses a projected coordinate system (meters), not geographic (lat/lon), for accurate distance calculations in DBSCAN.
* **Species Mapping:** Check the `SPECIES_MAP` dictionary at the top of the scripts. If your dataset has different species IDs, update this map before running.

## Citation

* **in work:** Multi-Source LiDAR Quantifies the Built Environment as a Deterministic Constraint on Tree Carbon Growth

## 🛠️ Installation

### Prerequisites
* Python 3.8+
* Anaconda (recommended)

### Dependencies
Install the required libraries using pip:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels scipy joblib
