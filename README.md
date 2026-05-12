# F2 EM Structure Function — ML Portfolio

A self-contained machine-learning project that models and extrapolates the
**proton electromagnetic structure function F<sub>2</sub>(x, Q<sup>2</sup>)** into the low-x region
using a full suite of supervised and unsupervised methods.

## Physics background

F<sub>2</sub> is measured in Deep Inelastic Scattering (DIS) experiments (H1, ZEUS, NMC, SLAC, BCDMS).
It depends on two kinematic variables:

* **x** — Bjorken scaling variable (parton momentum fraction)
* **Q<sup>2</sup>** — photon virtuality (resolution scale)

At very small x (x < 10<sup>-3</sup>) experimental data are sparse. All models are
trained on existing data and extrapolated into this low-x region.

## Project structure

```
proton_structure_ml/
├── src/
│   ├── data_loader.py     # data loading & feature utilities
│   ├── features.py        # feature engineering helpers
│   └── visualization.py   # shared plotting functions
├── notebooks/
│   ├── 00_EDA.ipynb               # Exploratory data analysis
│   ├── 01_regression.ipynb        # Linear, Ridge, Lasso, Polynomial
│   ├── 02_gpr.ipynb               # Gaussian Process Regression
│   ├── 03_tree_models.ipynb       # Random Forest & XGBoost
│   ├── 04_unsupervised.ipynb      # PCA, t-SNE, K-Means, DBSCAN, Isolation Forest
│   ├── 05_autoencoder.ipynb       # Bottleneck neural network
│   └── 06_model_comparison.ipynb  # Side-by-side comparison & physics discussion
├── results/
│   ├── figures/
│   └── metrics_comparison.csv
├── requirements.txt
└── create_notebooks.py   # run once to regenerate all notebooks
```

## Supervised methods

| Notebook | Methods |
|---|---|
| 01 | Linear Regression, Ridge (L2), Lasso (L1), Polynomial (deg 2 & 3) |
| 02 | Gaussian Process Regression — RBF kernel, Matérn kernel, uncertainty bands |
| 03 | Random Forest, XGBoost, feature importance |
| 05 | Bottleneck neural network (encoder–decoder architecture) |

## Unsupervised methods

| Notebook | Methods |
|---|---|
| 04 | PCA, t-SNE, K-Means, DBSCAN, Isolation Forest |

## Quickstart

```bash
git clone <your-repo>
cd proton_structure_ml
pip install -r requirements.txt

# (optional) regenerate notebooks
python create_notebooks.py

# launch Jupyter from the project root
jupyter lab
```

> **Data path**: set `DATA_DIR` at the top of each notebook to point to
> your local copy of the LeptonDIS `.dat` files.
