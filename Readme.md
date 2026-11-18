# Biodiesel Engine Optimization using GPR, NSGA-III, and MOPSO

Machine learning surrogate modeling and many-objective evolutionary optimization for diesel engine performance and emissions across varying engine speeds and biodiesel blend ratios.

## Overview

This repository trains **Gaussian Process Regression (GPR)** surrogate models on experimental engine data to map **Engine Speed (`RPM`)** and **Biodiesel Blend (`%`)** across seven engine performance and emission responses. It then applies **NSGA-III** and **MOPSO** to find Pareto-optimal operating conditions, ranks solutions via a weighted **Composite Desirability Function**, and benchmarks multi-seed convergence using Hypervolume (HV) and IGD.

## Objectives & Decision Variables

* **Inputs:** Engine Speed ($900\text{–}2200\text{ rpm}$) and Biodiesel Blend ($0\%\text{–}45\%$, `B0`–`B45`).
* **Minimize:** `BSFC` ($\text{kg/kWh}$, $w=3$), `NOx` ($\text{ppm}$, $w=1$), `CO` ($\text{ppm}$, $w=1$), `CO2` ($\%\text{ vol}$, $w=1$).
* **Maximize:** `BTE` ($\%$, $w=3$), `POWER` ($\text{W}$, $w=3$), `TORQUE` ($\text{Nm}$, $w=2$).

*(Note: `NSGA 3 SCRIPT.py` also includes a 4-objective benchmark comparing NSGA-III, MOPSO, and grid search across 10 seeds using brake-specific emissions $\text{NO}_{x,\text{sp}}$ and $\text{CO}_{\text{sp}}$ in $\text{g/kWh}$.)*

## Repository Structure

```text
├── NSGA 3 SCRIPT.py             # GPR training, LOO/LOBO CV, 7-obj NSGA-III, & 4-obj benchmark
├── NSGA 3 Auxilary Graphs.py    # Parity, CV, radar, parallel coordinate, & 3D plots for NSGA-III
├── MOPSO SCRIPT.py              # GPR training, LOO/LOBO CV, & 7-obj MOPSO optimizer
├── MOPSO Auxilary Graphs.py     # Parity, CV, radar, parallel coordinate, & 3D plots for MOPSO
└── README.md
```

## Quick Start

### 1. Install Dependencies
```bash
pip install numpy pandas matplotlib seaborn scikit-learn openpyxl plotly pymoo
```

### 2. Set Dataset Path
Update the `filepath` variable in `NSGA 3 SCRIPT.py` and `MOPSO SCRIPT.py` to point to your `.xlsx` dataset (columns: `SPEED (RPM)`, `BLEND (%)`, `BSFC`, `BTE`, `NOx`, `CO`, `POWER`, `TORQUE`, `CO2`):
```python
filepath = r"path/to/Organized Data.xlsx"
```

### 3. Run Optimization & Graphing Scripts
Run the main scripts first to generate the intermediate `.csv` and `.xlsx` files, then run the auxiliary scripts for publication figures:
```bash
python "NSGA 3 SCRIPT.py"
python "NSGA 3 Auxilary Graphs.py"

python "MOPSO SCRIPT.py"
python "MOPSO Auxilary Graphs.py"
```

## Key Outputs

* **Optimization Results:** `NSGA3_GPR_Full_Results.xlsx`, `MOPSO_GPR_Full_Results.xlsx`, and `table10.csv` / `table11.csv` (multi-seed HV, IGD, and weight sensitivity).
* **Model Validation:** `gpr_table2.csv` (Leave-One-Out and Leave-One-Blend-Out cross-validation metrics benchmarked against Quadratic RSM).
* **Figures (`results_figures_*` & `publication_plots_*`):** Parity and residual grids, convergence curves, desirability surfaces, sensitivity plots, radar comparisons, parallel coordinates, and interactive 3D Pareto fronts (`pareto_3d_interactive.html`).