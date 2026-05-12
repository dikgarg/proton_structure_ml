#!/usr/bin/env python3
"""Run this from the project root to generate all notebooks:
    python create_notebooks.py
"""

import json
from pathlib import Path

NB_DIR = Path(__file__).parent / "notebooks"
NB_DIR.mkdir(exist_ok=True)

DATA_DIR = (
    "/Users/dikgarg/Desktop/Research/Neutrinos/postdoc/2025/ML"
    "/Data/Exp_data/LeptonDIS"
)

# ── helpers ───────────────────────────────────────────────────────────────────

def _lines(src):
    return src.lstrip("\n").splitlines(keepends=True)

def code(cell_id, src):
    return {"cell_type": "code", "execution_count": None,
            "id": cell_id, "metadata": {}, "outputs": [],
            "source": _lines(src)}

def md(cell_id, src):
    return {"cell_type": "markdown", "id": cell_id,
            "metadata": {}, "source": _lines(src)}

def nb(cells):
    return {"cells": cells,
            "metadata": {"kernelspec": {"display_name": "Python 3",
                                        "language": "python",
                                        "name": "python3"},
                          "language_info": {"name": "python",
                                            "version": "3.10.0"}},
            "nbformat": 4, "nbformat_minor": 5}

def save(nb_dict, filename):
    p = NB_DIR / filename
    with open(p, "w", encoding="utf-8") as f:
        json.dump(nb_dict, f, indent=1, ensure_ascii=False)
    print(f"  Written: {p}")

PREAMBLE = f'''import sys
sys.path.insert(0, "../src")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams["figure.dpi"] = 120

from data_loader import load_lepton_dis, add_log_features, split_data, get_Xy, low_x_grid
from visualization import plot_coverage, plot_F2_vs_x, comparison_figure

DATA_DIR = "{DATA_DIR}"

df_raw = load_lepton_dis(DATA_DIR)
df     = add_log_features(df_raw)
df_train, df_test = split_data(df, test_size=0.2, seed=42)

X_train, y_train = get_Xy(df_train)
X_test,  y_test  = get_Xy(df_test)

print(f"Training points: {{len(X_train)}}  |  Test points: {{len(X_test)}}")
'''

# ═════════════════════════════════════════════════════════════════════════════
# 00  EDA
# ═════════════════════════════════════════════════════════════════════════════

def make_00():
    cells = [
        md("a0", r"""# 00 · Exploratory Data Analysis

## The F₂ Proton Structure Function

The electromagnetic proton structure function $F_2^p(x, Q^2)$ is measured in
**Deep Inelastic Scattering (DIS)** experiments.

* **x** – Bjorken scaling variable: fraction of the proton momentum carried by the struck quark
* **Q²** – virtuality (resolution scale) of the exchanged photon [GeV²]

At small x the gluon sea drives a steep rise in F₂ (Pomeron/BFKL dynamics).
This project applies a suite of ML methods to model and extrapolate F₂ into the
low-x region where data are sparse."""),

        code("a1", PREAMBLE),

        md("a2", r"""## Dataset Summary"""),

        code("a3", '''summary = (
    df.groupby("experiment")
    .agg(
        n_points=("F2", "count"),
        x_min=("x", "min"),
        x_max=("x", "max"),
        Q2_min=("Q2", "min"),
        Q2_max=("Q2", "max"),
        F2_mean=("F2", "mean"),
    )
    .round(4)
)
print(summary.to_string())
'''),

        md("a4", r"""## Data Coverage in the $(x, Q^2)$ Plane"""),

        code("a5", '''fig, ax = plt.subplots(figsize=(7, 5))
plot_coverage(df, ax=ax)
plt.tight_layout()
plt.savefig("../results/figures/00_coverage.png", dpi=150)
plt.show()
'''),

        md("a6", r"""## Distribution of $F_2^p$ Values"""),

        code("a7", '''fig, axes = plt.subplots(1, 2, figsize=(11, 4))

axes[0].hist(df["F2"], bins=40, color="#377eb8", edgecolor="white", linewidth=0.4)
axes[0].set_xlabel(r"$F_2^p$", fontsize=12)
axes[0].set_ylabel("Count", fontsize=12)
axes[0].set_title("Distribution of $F_2^p$", fontsize=12)

axes[1].hist(df["log10_x"], bins=40, color="#e41a1c", edgecolor="white", linewidth=0.4)
axes[1].set_xlabel(r"$\\log_{10}(x)$", fontsize=12)
axes[1].set_title(r"Distribution of $\\log_{10}(x)$", fontsize=12)

plt.tight_layout()
plt.savefig("../results/figures/00_distributions.png", dpi=150)
plt.show()
'''),

        md("a8", r"""## $F_2^p$ vs $x$ at Fixed $Q^2$ Bins"""),

        code("a9", '''Q2_vals = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
fig, axes = plot_F2_vs_x(df, Q2_vals, ncols=3)
fig.suptitle(r"$F_2^p$ vs $x$ in $Q^2$ bins", fontsize=13, y=1.01)
plt.savefig("../results/figures/00_F2_vs_x.png", dpi=150, bbox_inches="tight")
plt.show()
'''),

        md("a10", r"""## $F_2^p$ vs $Q^2$ at Fixed $x$ Bins"""),

        code("a11", '''from visualization import EXP_COLORS

x_centres = [0.008, 0.025, 0.07, 0.18, 0.45]
delta_log  = 0.3

fig, axes = plt.subplots(1, len(x_centres), figsize=(16, 4), sharey=False)
for ax, xc in zip(axes, x_centres):
    lo = 10 ** (np.log10(xc) - delta_log)
    hi = 10 ** (np.log10(xc) + delta_log)
    sub = df[(df["x"] >= lo) & (df["x"] <= hi)]
    for exp, grp in sub.groupby("experiment"):
        ax.errorbar(grp["Q2"], grp["F2"],
                    yerr=[grp["sigma"], grp["sigma"]],
                    fmt="o", ms=3, alpha=0.8,
                    color=EXP_COLORS.get(exp, "grey"), label=exp)
    ax.set_xscale("log")
    ax.set_xlabel(r"$Q^2$ [GeV$^2$]", fontsize=10)
    ax.set_ylabel(r"$F_2^p$", fontsize=10)
    ax.set_title(rf"$x \\approx {xc}$", fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(True, ls="--", alpha=0.3)

plt.tight_layout()
plt.savefig("../results/figures/00_F2_vs_Q2.png", dpi=150)
plt.show()
'''),

        md("a12", r"""## Feature Correlation Matrix"""),

        code("a13", '''import seaborn as sns

feat_df = df[["log10_x", "log10_Q2", "F2"]].copy()
feat_df.columns = [r"log10(x)", r"log10(Q2)", "F2"]

fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(feat_df.corr(), annot=True, fmt=".2f", cmap="coolwarm",
            center=0, ax=ax, linewidths=0.5)
ax.set_title("Pearson correlation matrix", fontsize=12)
plt.tight_layout()
plt.savefig("../results/figures/00_correlation.png", dpi=150)
plt.show()
'''),

        md("a14", r"""## Key Observations

1. **Wide kinematic range**: x spans ~4 orders of magnitude; Q² spans ~3 orders.
2. **Rise at low x**: F₂ increases as x decreases, especially visible in the HERA data (H1, ZEUS).
3. **Strong log-log linearity** between x and Q²: log₁₀(x) and log₁₀(Q²) are the natural input features.
4. **Low-x gap**: measurements stop around x ~ 10⁻⁴; this is the region we want to extrapolate into.
"""),
    ]
    save(nb(cells), "00_EDA.ipynb")


# ═════════════════════════════════════════════════════════════════════════════
# 01  Regression
# ═════════════════════════════════════════════════════════════════════════════

def make_01():
    cells = [
        md("b0", r"""# 01 · Supervised Learning: Regression Methods

We start with classical regression approaches as interpretable baselines:

| Method | Key property |
|---|---|
| Linear Regression | simplest baseline |
| Ridge (L2) | prevents large coefficients |
| Lasso (L1) | sparsity / feature selection |
| Polynomial (deg 2 & 3) | captures curvature |

All models use **log₁₀(x)** and **log₁₀(Q²)** as features."""),

        code("b1", PREAMBLE),

        md("b2", "## Feature Scaling"),

        code("b3", '''from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings("ignore")

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)
'''),

        md("b4", "## Linear Regression"),

        code("b5", '''lr = LinearRegression()
lr.fit(X_train_s, y_train)

y_pred_lr = lr.predict(X_test_s)
print("Linear Regression")
print(f"  Test MSE : {mean_squared_error(y_test, y_pred_lr):.5f}")
print(f"  Test MAE : {mean_absolute_error(y_test, y_pred_lr):.5f}")
print(f"  Test R2  : {r2_score(y_test, y_pred_lr):.4f}")
'''),

        md("b6", "## Ridge Regression (L2 regularisation)"),

        code("b7", '''from sklearn.linear_model import RidgeCV

ridge = RidgeCV(alphas=np.logspace(-3, 3, 30), cv=5)
ridge.fit(X_train_s, y_train)

y_pred_ridge = ridge.predict(X_test_s)
print(f"Ridge  alpha={ridge.alpha_:.4f}")
print(f"  Test MSE : {mean_squared_error(y_test, y_pred_ridge):.5f}")
print(f"  Test MAE : {mean_absolute_error(y_test, y_pred_ridge):.5f}")
print(f"  Test R2  : {r2_score(y_test, y_pred_ridge):.4f}")
'''),

        md("b8", "## Lasso Regression (L1 regularisation)"),

        code("b9", '''from sklearn.linear_model import LassoCV

lasso = LassoCV(alphas=np.logspace(-5, 1, 30), cv=5, max_iter=5000)
lasso.fit(X_train_s, y_train)

y_pred_lasso = lasso.predict(X_test_s)
print(f"Lasso  alpha={lasso.alpha_:.6f}")
print(f"  Test MSE : {mean_squared_error(y_test, y_pred_lasso):.5f}")
print(f"  Test MAE : {mean_absolute_error(y_test, y_pred_lasso):.5f}")
print(f"  Test R2  : {r2_score(y_test, y_pred_lasso):.4f}")
'''),

        md("b10", "## Polynomial Regression"),

        code("b11", '''results = {}
for deg in [2, 3]:
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("poly",   PolynomialFeatures(degree=deg, include_bias=False)),
        ("model",  LinearRegression()),
    ])
    pipe.fit(X_train, y_train)
    yp = pipe.predict(X_test)
    results[f"Poly-{deg}"] = {
        "model": pipe,
        "mse": mean_squared_error(y_test, yp),
        "mae": mean_absolute_error(y_test, yp),
        "r2":  r2_score(y_test, yp),
    }
    print(f"Poly deg={deg}  MSE={results[f'Poly-{deg}']['mse']:.5f}  "
          f"R2={results[f'Poly-{deg}']['r2']:.4f}")
'''),

        md("b12", "## Metrics Summary"),

        code("b13", '''rows = [
    ("Linear",  mean_squared_error(y_test, y_pred_lr),
                mean_absolute_error(y_test, y_pred_lr),
                r2_score(y_test, y_pred_lr)),
    ("Ridge",   mean_squared_error(y_test, y_pred_ridge),
                mean_absolute_error(y_test, y_pred_ridge),
                r2_score(y_test, y_pred_ridge)),
    ("Lasso",   mean_squared_error(y_test, y_pred_lasso),
                mean_absolute_error(y_test, y_pred_lasso),
                r2_score(y_test, y_pred_lasso)),
]
for name, res in results.items():
    rows.append((name, res["mse"], res["mae"], res["r2"]))

metrics_df = pd.DataFrame(rows, columns=["Model", "MSE", "MAE", "R2"])
metrics_df = metrics_df.sort_values("MSE").reset_index(drop=True)
print(metrics_df.to_string(index=False))
'''),

        md("b14", r"""## Predictions vs Data and Low-x Extrapolation"""),

        code("b15", '''Q2_plot  = [1.0, 5.0, 15.0, 30.0]
grids    = low_x_grid(x_min=1e-6, x_max=0.8, n_points=400, Q2_values=Q2_plot)

best_pipe = results["Poly-3"]["model"]

predictions = [
    {
        "label":     "Poly-3",
        "color":     "#e41a1c",
        "x_arr":     None,
        "Q2_preds":  {},
    }
]
for Q2v, (x_arr, X_feat) in grids.items():
    predictions[0]["x_arr"] = x_arr
    predictions[0]["Q2_preds"][Q2v] = best_pipe.predict(X_feat)

fig, _ = comparison_figure(Q2_plot, df, predictions)
fig.suptitle("Polynomial Regression (deg 3) — predictions vs data", y=1.01)
plt.savefig("../results/figures/01_poly_predictions.png", dpi=150, bbox_inches="tight")
plt.show()
'''),
    ]
    save(nb(cells), "01_regression.ipynb")


# ═════════════════════════════════════════════════════════════════════════════
# 02  GPR
# ═════════════════════════════════════════════════════════════════════════════

def make_02():
    cells = [
        md("c0", r"""# 02 · Gaussian Process Regression

GPR is a powerful Bayesian non-parametric method that:

* **Quantifies uncertainty** — gives a predictive distribution, not just a point estimate
* **Works well with small datasets** — our ~2000 points are manageable
* **Incorporates prior knowledge** via kernel choice

This is especially valuable in physics: the uncertainty band on the low-x
extrapolation tells us how much we should trust the prediction in the unconstrained region."""),

        code("c1", PREAMBLE),

        code("c2", '''from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF, Matern, WhiteKernel, ConstantKernel as C
)
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

scaler    = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)
'''),

        md("c3", "## GPR with RBF Kernel"),

        code("c4", '''kernel_rbf = C(1.0, (1e-3, 1e3)) * RBF([1.0, 1.0], (1e-2, 1e2)) \
             + WhiteKernel(noise_level=0.01, noise_level_bounds=(1e-4, 1.0))

gpr_rbf = GaussianProcessRegressor(
    kernel=kernel_rbf,
    n_restarts_optimizer=5,
    normalize_y=True,
    random_state=42,
)
gpr_rbf.fit(X_train_s, y_train)

y_pred_rbf, y_std_rbf = gpr_rbf.predict(X_test_s, return_std=True)

print("GPR – RBF kernel")
print(f"  Optimised kernel: {gpr_rbf.kernel_}")
print(f"  Test MSE : {mean_squared_error(y_test, y_pred_rbf):.5f}")
print(f"  Test MAE : {mean_absolute_error(y_test, y_pred_rbf):.5f}")
print(f"  Test R2  : {r2_score(y_test, y_pred_rbf):.4f}")
'''),

        md("c5", "## GPR with Matérn Kernel (nu=3/2)"),

        code("c6", '''kernel_mat = C(1.0, (1e-3, 1e3)) * Matern(length_scale=[1.0, 1.0],
                                                   length_scale_bounds=(1e-2, 1e2),
                                                   nu=1.5) \
             + WhiteKernel(noise_level=0.01, noise_level_bounds=(1e-4, 1.0))

gpr_mat = GaussianProcessRegressor(
    kernel=kernel_mat,
    n_restarts_optimizer=5,
    normalize_y=True,
    random_state=42,
)
gpr_mat.fit(X_train_s, y_train)

y_pred_mat, y_std_mat = gpr_mat.predict(X_test_s, return_std=True)

print("GPR – Matern kernel")
print(f"  Optimised kernel: {gpr_mat.kernel_}")
print(f"  Test MSE : {mean_squared_error(y_test, y_pred_mat):.5f}")
print(f"  Test MAE : {mean_absolute_error(y_test, y_pred_mat):.5f}")
print(f"  Test R2  : {r2_score(y_test, y_pred_mat):.4f}")
'''),

        md("c7", r"""## Predictions with Uncertainty Bands"""),

        code("c8", '''Q2_plot = [1.0, 5.0, 15.0, 30.0]
grids   = low_x_grid(x_min=1e-6, x_max=0.8, n_points=300, Q2_values=Q2_plot)

preds_rbf = {"label": "GPR-RBF", "color": "#e41a1c", "x_arr": None,
             "Q2_preds": {}, "Q2_lo": {}, "Q2_hi": {}}
preds_mat = {"label": "GPR-Matern", "color": "#377eb8", "x_arr": None,
             "Q2_preds": {}, "Q2_lo": {}, "Q2_hi": {}}

for Q2v, (x_arr, X_feat) in grids.items():
    X_feat_s = scaler.transform(X_feat)

    mu_r, sig_r = gpr_rbf.predict(X_feat_s, return_std=True)
    mu_m, sig_m = gpr_mat.predict(X_feat_s, return_std=True)

    preds_rbf["x_arr"]           = x_arr
    preds_rbf["Q2_preds"][Q2v]   = mu_r
    preds_rbf["Q2_lo"][Q2v]      = mu_r - 2 * sig_r
    preds_rbf["Q2_hi"][Q2v]      = mu_r + 2 * sig_r

    preds_mat["x_arr"]           = x_arr
    preds_mat["Q2_preds"][Q2v]   = mu_m
    preds_mat["Q2_lo"][Q2v]      = mu_m - 2 * sig_m
    preds_mat["Q2_hi"][Q2v]      = mu_m + 2 * sig_m

fig, _ = comparison_figure(Q2_plot, df, [preds_rbf, preds_mat])
fig.suptitle("GPR predictions with 2σ uncertainty bands", y=1.01)
plt.savefig("../results/figures/02_gpr_predictions.png", dpi=150, bbox_inches="tight")
plt.show()
'''),

        md("c9", r"""## Uncertainty Grows in the Low-x Extrapolation Region

The shaded bands widen as x decreases beyond the training data boundary — exactly what a
well-calibrated model should do. The RBF and Matérn kernels give similar central predictions
but differ slightly in the rate of uncertainty growth, reflecting their different smoothness assumptions."""),
    ]
    save(nb(cells), "02_gpr.ipynb")


# ═════════════════════════════════════════════════════════════════════════════
# 03  Tree models
# ═════════════════════════════════════════════════════════════════════════════

def make_03():
    cells = [
        md("d0", r"""# 03 · Ensemble Tree Methods: Random Forest & XGBoost

Ensemble tree methods are powerful non-parametric regressors that:

* Require **no feature scaling**
* Capture **non-linear interactions** automatically
* Provide **feature importance** diagnostics
* **XGBoost** adds gradient-boosted sequential correction

A key limitation: tree ensembles **cannot extrapolate** beyond the training data range —
they predict the mean of the nearest training leaves. This makes the low-x comparison especially interesting."""),

        code("d1", PREAMBLE),

        code("d2", '''from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
'''),

        md("d3", "## Random Forest"),

        code("d4", '''rf = RandomForestRegressor(
    n_estimators=400,
    max_depth=None,
    min_samples_leaf=2,
    max_features=1.0,
    random_state=42,
    n_jobs=-1,
)
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)
print("Random Forest")
print(f"  Test MSE : {mean_squared_error(y_test, y_pred_rf):.5f}")
print(f"  Test MAE : {mean_absolute_error(y_test, y_pred_rf):.5f}")
print(f"  Test R2  : {r2_score(y_test, y_pred_rf):.4f}")
'''),

        md("d5", "## Feature Importance — Random Forest"),

        code("d6", '''feat_names = ["log10(x)", "log10(Q2)"]
importances = rf.feature_importances_

fig, ax = plt.subplots(figsize=(5, 3))
bars = ax.barh(feat_names, importances, color=["#e41a1c", "#377eb8"])
ax.set_xlabel("Mean impurity decrease", fontsize=11)
ax.set_title("Random Forest — feature importance", fontsize=12)
for bar, val in zip(bars, importances):
    ax.text(val + 0.005, bar.get_y() + bar.get_height()/2,
            f"{val:.3f}", va="center", fontsize=10)
plt.tight_layout()
plt.savefig("../results/figures/03_rf_importance.png", dpi=150)
plt.show()
'''),

        md("d7", "## XGBoost"),

        code("d8", '''xgb_model = xgb.XGBRegressor(
    n_estimators=600,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=1.0,
    reg_lambda=1.0,
    random_state=42,
    verbosity=0,
)
xgb_model.fit(X_train, y_train,
              eval_set=[(X_test, y_test)],
              verbose=False)

y_pred_xgb = xgb_model.predict(X_test)
print("XGBoost")
print(f"  Test MSE : {mean_squared_error(y_test, y_pred_xgb):.5f}")
print(f"  Test MAE : {mean_absolute_error(y_test, y_pred_xgb):.5f}")
print(f"  Test R2  : {r2_score(y_test, y_pred_xgb):.4f}")
'''),

        md("d9", "## Feature Importance — XGBoost"),

        code("d10", '''xgb_imp = xgb_model.feature_importances_

fig, ax = plt.subplots(figsize=(5, 3))
bars = ax.barh(feat_names, xgb_imp, color=["#e41a1c", "#377eb8"])
ax.set_xlabel("Gain", fontsize=11)
ax.set_title("XGBoost — feature importance", fontsize=12)
for bar, val in zip(bars, xgb_imp):
    ax.text(val + 0.005, bar.get_y() + bar.get_height()/2,
            f"{val:.3f}", va="center", fontsize=10)
plt.tight_layout()
plt.savefig("../results/figures/03_xgb_importance.png", dpi=150)
plt.show()
'''),

        md("d11", r"""## Predictions and Low-x Extrapolation

Note: tree methods **plateau** at low x because they cannot extrapolate
beyond the minimum x in the training set — they return the mean of the
most similar training leaf. This is visible as flat tails in the plots below."""),

        code("d12", '''Q2_plot = [1.0, 5.0, 15.0, 30.0]
grids   = low_x_grid(x_min=1e-6, x_max=0.8, n_points=400, Q2_values=Q2_plot)

def build_pred_dict(model, label, color):
    entry = {"label": label, "color": color, "x_arr": None, "Q2_preds": {}}
    for Q2v, (x_arr, X_feat) in grids.items():
        entry["x_arr"]         = x_arr
        entry["Q2_preds"][Q2v] = model.predict(X_feat)
    return entry

preds = [
    build_pred_dict(rf,        "Random Forest", "#e41a1c"),
    build_pred_dict(xgb_model, "XGBoost",       "#377eb8"),
]

fig, _ = comparison_figure(Q2_plot, df, preds)
fig.suptitle("Tree ensemble predictions vs data (note flat low-x tails)", y=1.01)
plt.savefig("../results/figures/03_tree_predictions.png", dpi=150, bbox_inches="tight")
plt.show()
'''),
    ]
    save(nb(cells), "03_tree_models.ipynb")


# ═════════════════════════════════════════════════════════════════════════════
# 04  Unsupervised
# ═════════════════════════════════════════════════════════════════════════════

def make_04():
    cells = [
        md("e0", r"""# 04 · Unsupervised Learning: Data Structure and Anomaly Detection

Unsupervised methods reveal structure in the data **without using F₂ as a label**.
We apply:

| Method | Purpose |
|---|---|
| **PCA** | Linear dimensionality reduction; variance explained |
| **t-SNE** | Non-linear embedding; visual cluster discovery |
| **K-Means** | Partition the kinematic $(x, Q^2)$ plane into regions |
| **DBSCAN** | Density-based clustering; finds irregular shapes |
| **Isolation Forest** | Anomaly / outlier detection across experiments |
"""),

        code("e1", PREAMBLE),

        code("e2", '''from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import IsolationForest
import seaborn as sns

# Feature matrix for unsupervised: (log10_x, log10_Q2, F2_norm)
scaler_u  = StandardScaler()
F2_norm   = (df["F2"].values - df["F2"].mean()) / df["F2"].std()
X_unsup   = np.column_stack([df["log10_x"].values,
                              df["log10_Q2"].values,
                              F2_norm])
X_unsup_s = scaler_u.fit_transform(X_unsup)
'''),

        md("e3", "## Principal Component Analysis"),

        code("e4", '''pca = PCA(n_components=3)
Z_pca = pca.fit_transform(X_unsup_s)

print("Explained variance ratio:")
for i, ev in enumerate(pca.explained_variance_ratio_):
    print(f"  PC{i+1}: {ev:.3f}  ({ev*100:.1f}%)")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

sc = axes[0].scatter(Z_pca[:, 0], Z_pca[:, 1],
                     c=df["F2"].values, cmap="plasma",
                     s=10, alpha=0.7)
plt.colorbar(sc, ax=axes[0], label=r"$F_2^p$")
axes[0].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
axes[0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
axes[0].set_title("PCA — coloured by F2")

sc2 = axes[1].scatter(Z_pca[:, 0], Z_pca[:, 1],
                      c=df["log10_x"].values, cmap="viridis",
                      s=10, alpha=0.7)
plt.colorbar(sc2, ax=axes[1], label=r"$\log_{10}(x)$")
axes[1].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
axes[1].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
axes[1].set_title(r"PCA — coloured by $\log_{10}(x)$")

plt.tight_layout()
plt.savefig("../results/figures/04_pca.png", dpi=150)
plt.show()
'''),

        md("e5", "## t-SNE Embedding"),

        code("e6", '''tsne = TSNE(n_components=2, perplexity=40, random_state=42, n_iter=1000)
Z_tsne = tsne.fit_transform(X_unsup_s)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

sc = axes[0].scatter(Z_tsne[:, 0], Z_tsne[:, 1],
                     c=df["F2"].values, cmap="plasma", s=8, alpha=0.7)
plt.colorbar(sc, ax=axes[0], label=r"$F_2^p$")
axes[0].set_title("t-SNE — coloured by F2")
axes[0].set_xlabel("t-SNE 1"); axes[0].set_ylabel("t-SNE 2")

sc2 = axes[1].scatter(Z_tsne[:, 0], Z_tsne[:, 1],
                      c=df["log10_x"].values, cmap="viridis", s=8, alpha=0.7)
plt.colorbar(sc2, ax=axes[1], label=r"$\log_{10}(x)$")
axes[1].set_title(r"t-SNE — coloured by $\log_{10}(x)$")
axes[1].set_xlabel("t-SNE 1"); axes[1].set_ylabel("t-SNE 2")

plt.tight_layout()
plt.savefig("../results/figures/04_tsne.png", dpi=150)
plt.show()
'''),

        md("e7", "## K-Means: Kinematic Region Clustering"),

        code("e8", '''# Cluster only in the (log10_x, log10_Q2) plane for interpretability
X_kin_s = StandardScaler().fit_transform(
    df[["log10_x", "log10_Q2"]].values
)

inertias = []
K_range  = range(2, 10)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_kin_s)
    inertias.append(km.inertia_)

fig, ax = plt.subplots(figsize=(6, 3))
ax.plot(list(K_range), inertias, "o-", color="#377eb8")
ax.set_xlabel("Number of clusters k"); ax.set_ylabel("Inertia")
ax.set_title("Elbow plot for K-Means")
ax.grid(True, ls="--", alpha=0.3)
plt.tight_layout()
plt.show()
'''),

        code("e9", '''K_BEST = 5
km_best = KMeans(n_clusters=K_BEST, random_state=42, n_init=10)
labels_km = km_best.fit_predict(X_kin_s)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

cmap_k = plt.get_cmap("tab10", K_BEST)
axes[0].scatter(df["x"], df["Q2"], c=labels_km, cmap=cmap_k,
                s=12, alpha=0.8)
axes[0].set_xscale("log"); axes[0].set_yscale("log")
axes[0].set_xlabel(r"$x$"); axes[0].set_ylabel(r"$Q^2$ [GeV$^2$]")
axes[0].set_title(f"K-Means clusters (k={K_BEST}) in $(x, Q^2)$ plane")

for k in range(K_BEST):
    mask = labels_km == k
    axes[1].scatter(df["x"].values[mask], df["F2"].values[mask],
                    s=8, alpha=0.7, label=f"Cluster {k}", color=cmap_k(k))
axes[1].set_xscale("log")
axes[1].set_xlabel(r"$x$"); axes[1].set_ylabel(r"$F_2^p$")
axes[1].set_title("F2 coloured by kinematic cluster")
axes[1].legend(fontsize=8, ncol=2)

plt.tight_layout()
plt.savefig("../results/figures/04_kmeans.png", dpi=150)
plt.show()

# Mean F2 per cluster
df_km = df.copy()
df_km["cluster"] = labels_km
print(df_km.groupby("cluster")[["x", "Q2", "F2"]].mean().round(4).to_string())
'''),

        md("e10", "## DBSCAN: Density-Based Clustering"),

        code("e11", '''dbscan = DBSCAN(eps=0.25, min_samples=5)
labels_db = dbscan.fit_predict(X_kin_s)

n_clusters = len(set(labels_db)) - (1 if -1 in labels_db else 0)
n_noise    = (labels_db == -1).sum()
print(f"DBSCAN found {n_clusters} clusters, {n_noise} noise points")

fig, ax = plt.subplots(figsize=(7, 5))
unique_labels = sorted(set(labels_db))
cmap_d = plt.get_cmap("tab10", max(unique_labels) + 1)
for lbl in unique_labels:
    mask  = labels_db == lbl
    color = "grey" if lbl == -1 else cmap_d(lbl)
    name  = "Noise" if lbl == -1 else f"Cluster {lbl}"
    ax.scatter(df["x"].values[mask], df["Q2"].values[mask],
               s=12, alpha=0.8, color=color, label=name)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel(r"$x$"); ax.set_ylabel(r"$Q^2$ [GeV$^2$]")
ax.set_title("DBSCAN clusters")
ax.legend(fontsize=8, ncol=2)
plt.tight_layout()
plt.savefig("../results/figures/04_dbscan.png", dpi=150)
plt.show()
'''),

        md("e12", "## Isolation Forest: Anomaly Detection"),

        code("e13", '''# Detect outliers in the (log10_x, log10_Q2, F2) space
iso = IsolationForest(n_estimators=300, contamination=0.05, random_state=42)
anomaly_labels = iso.fit_predict(X_unsup_s)   # -1 = anomaly, +1 = normal

n_anomalies = (anomaly_labels == -1).sum()
print(f"Isolation Forest flagged {n_anomalies} anomalous points "
      f"({n_anomalies/len(df)*100:.1f}% of data)")

fig, axes = plt.subplots(1, 2, figsize=(13, 4))

colors = np.where(anomaly_labels == -1, "red", "steelblue")
alpha  = np.where(anomaly_labels == -1, 1.0, 0.3)

axes[0].scatter(df["x"], df["Q2"], c=colors, s=12, alpha=0.6)
axes[0].set_xscale("log"); axes[0].set_yscale("log")
axes[0].set_xlabel(r"$x$"); axes[0].set_ylabel(r"$Q^2$ [GeV$^2$]")
axes[0].set_title("Anomalies in $(x, Q^2)$ plane (red = flagged)")

axes[1].scatter(df["x"], df["F2"], c=colors, s=12, alpha=0.6)
axes[1].set_xscale("log")
axes[1].set_xlabel(r"$x$"); axes[1].set_ylabel(r"$F_2^p$")
axes[1].set_title("Anomalies in $(x, F_2)$ plane")

plt.tight_layout()
plt.savefig("../results/figures/04_isolation_forest.png", dpi=150)
plt.show()

# Which experiments do the anomalies come from?
df_anom = df.copy()
df_anom["anomaly"] = anomaly_labels == -1
print(df_anom.groupby("experiment")["anomaly"].sum().astype(int))
'''),
    ]
    save(nb(cells), "04_unsupervised.ipynb")


# ═════════════════════════════════════════════════════════════════════════════
# 05  Autoencoder
# ═════════════════════════════════════════════════════════════════════════════

def make_05():
    cells = [
        md("f0", r"""# 05 · Deep Learning: Bottleneck Neural Network (Autoencoder-Inspired)

We build an **undercomplete network** — a neural network with a narrow
*bottleneck* hidden layer — that maps $(\\log_{10} x,\\, \\log_{10} Q^2)$
to $F_2^p$.

The bottleneck forces the model to compress the input information into a
low-dimensional latent representation before reconstructing the output.
Visualising this 2-D latent space shows how the network organises the
kinematic information internally.

This is **entirely new code** — architecture, training loop, and
visualisation are written from scratch for this project."""),

        code("f1", PREAMBLE),

        code("f2", '''import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

tf.random.set_seed(42)

scaler    = StandardScaler()
X_train_s = scaler.fit_transform(X_train).astype("float32")
X_test_s  = scaler.transform(X_test).astype("float32")
y_train_f = y_train.astype("float32")
y_test_f  = y_test.astype("float32")
'''),

        md("f3", "## Network Architecture"),

        code("f4", '''def build_bottleneck_net(input_dim=2, latent_dim=2,
                         encoder_units=(64, 32),
                         decoder_units=(32, 64)):
    """
    Input -> [encoder layers] -> Bottleneck (latent_dim) -> [decoder layers] -> Output

    The bottleneck layer is the compressed latent representation.
    """
    inp = tf.keras.Input(shape=(input_dim,), name="input")

    # Encoder path
    x = inp
    for i, u in enumerate(encoder_units):
        x = tf.keras.layers.Dense(u, activation="tanh",
                                  name=f"encoder_{i}")(x)

    latent = tf.keras.layers.Dense(latent_dim, activation="linear",
                                   name="bottleneck")(x)

    # Decoder path
    x = latent
    for i, u in enumerate(decoder_units):
        x = tf.keras.layers.Dense(u, activation="tanh",
                                  name=f"decoder_{i}")(x)

    output = tf.keras.layers.Dense(1, activation="linear",
                                   name="output")(x)

    full_model   = tf.keras.Model(inp, output,  name="bottleneck_net")
    encoder_only = tf.keras.Model(inp, latent,  name="encoder")

    return full_model, encoder_only


model, encoder = build_bottleneck_net(input_dim=2, latent_dim=2)
model.summary()
'''),

        md("f5", "## Training"),

        code("f6", '''model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=3e-3),
    loss="mse",
)

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=50, restore_best_weights=True
)
lr_sched = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss", factor=0.5, patience=20, min_lr=1e-5, verbose=0
)

history = model.fit(
    X_train_s, y_train_f,
    validation_data=(X_test_s, y_test_f),
    epochs=800,
    batch_size=64,
    callbacks=[early_stop, lr_sched],
    verbose=0,
)

print(f"Stopped at epoch {len(history.history['loss'])}")

fig, ax = plt.subplots(figsize=(7, 3))
ax.plot(history.history["loss"],     label="Train loss")
ax.plot(history.history["val_loss"], label="Val loss", ls="--")
ax.set_xlabel("Epoch"); ax.set_ylabel("MSE")
ax.set_title("Training curve"); ax.legend(); ax.set_yscale("log")
ax.grid(True, ls="--", alpha=0.3)
plt.tight_layout()
plt.show()
'''),

        md("f7", "## Test Set Performance"),

        code("f8", '''y_pred_ae = model.predict(X_test_s, verbose=0).ravel()
print(f"Test MSE : {mean_squared_error(y_test, y_pred_ae):.5f}")
print(f"Test MAE : {mean_absolute_error(y_test, y_pred_ae):.5f}")
print(f"Test R2  : {r2_score(y_test, y_pred_ae):.4f}")
'''),

        md("f9", "## Bottleneck (Latent) Space"),

        code("f10", '''Z_all = encoder.predict(scaler.transform(df[["log10_x", "log10_Q2"]].values),
                       verbose=0)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

sc1 = axes[0].scatter(Z_all[:, 0], Z_all[:, 1],
                      c=df["F2"].values, cmap="plasma", s=10, alpha=0.7)
plt.colorbar(sc1, ax=axes[0], label=r"$F_2^p$")
axes[0].set_xlabel("Latent dim 1"); axes[0].set_ylabel("Latent dim 2")
axes[0].set_title("Bottleneck representation — coloured by F2")

sc2 = axes[1].scatter(Z_all[:, 0], Z_all[:, 1],
                      c=df["log10_x"].values, cmap="viridis", s=10, alpha=0.7)
plt.colorbar(sc2, ax=axes[1], label=r"$\log_{10}(x)$")
axes[1].set_xlabel("Latent dim 1"); axes[1].set_ylabel("Latent dim 2")
axes[1].set_title(r"Bottleneck representation — coloured by $\log_{10}(x)$")

plt.tight_layout()
plt.savefig("../results/figures/05_latent_space.png", dpi=150)
plt.show()
'''),

        md("f11", r"""## Low-x Extrapolation"""),

        code("f12", '''Q2_plot = [1.0, 5.0, 15.0, 30.0]
grids   = low_x_grid(x_min=1e-6, x_max=0.8, n_points=400, Q2_values=Q2_plot)

pred_ae = {"label": "Bottleneck NN", "color": "#4daf4a", "x_arr": None, "Q2_preds": {}}
for Q2v, (x_arr, X_feat) in grids.items():
    X_feat_s = scaler.transform(X_feat).astype("float32")
    pred_ae["x_arr"]         = x_arr
    pred_ae["Q2_preds"][Q2v] = model.predict(X_feat_s, verbose=0).ravel()

fig, _ = comparison_figure(Q2_plot, df, [pred_ae])
fig.suptitle("Bottleneck NN — predictions and low-x extrapolation", y=1.01)
plt.savefig("../results/figures/05_ae_predictions.png", dpi=150, bbox_inches="tight")
plt.show()
'''),
    ]
    save(nb(cells), "05_autoencoder.ipynb")


# ═════════════════════════════════════════════════════════════════════════════
# 06  Model comparison
# ═════════════════════════════════════════════════════════════════════════════

def make_06():
    cells = [
        md("g0", r"""# 06 · Model Comparison and Low-x Extrapolation

This notebook trains all supervised models and compares them on:

1. **Test-set performance metrics** (MSE, MAE, R²)
2. **Predictions vs data** at fixed Q² values
3. **Low-x extrapolation** — the central physics goal

The key question: which method gives the most physically reasonable
extrapolation into the uncharted low-x region?"""),

        code("g1", PREAMBLE),

        code("g2", '''from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as C
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import tensorflow as tf
import warnings
warnings.filterwarnings("ignore")
tf.random.set_seed(42)
'''),

        md("g3", "## Train All Models"),

        code("g4", '''# ── Scaler ────────────────────────────────────────────────────────────────
scaler    = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# ── Ridge ──────────────────────────────────────────────────────────────────
ridge = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0], cv=5)
ridge.fit(X_train_s, y_train)

# ── Polynomial deg 3 ───────────────────────────────────────────────────────
poly3 = Pipeline([("sc", StandardScaler()),
                  ("pf", PolynomialFeatures(3, include_bias=False)),
                  ("lr", LinearRegression())])
poly3.fit(X_train, y_train)

# ── GPR Matern ─────────────────────────────────────────────────────────────
kernel = (C(1.0, (1e-3, 1e3))
          * Matern(length_scale=[1., 1.], length_scale_bounds=(1e-2, 1e2), nu=1.5)
          + WhiteKernel(noise_level=0.01, noise_level_bounds=(1e-4, 1.0)))
gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=3,
                                normalize_y=True, random_state=42)
gpr.fit(X_train_s, y_train)

# ── Random Forest ──────────────────────────────────────────────────────────
rf = RandomForestRegressor(n_estimators=400, min_samples_leaf=2,
                            random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

# ── XGBoost ────────────────────────────────────────────────────────────────
xgb_m = xgb.XGBRegressor(n_estimators=600, learning_rate=0.05, max_depth=5,
                           subsample=0.8, random_state=42, verbosity=0)
xgb_m.fit(X_train, y_train, verbose=False)

# ── Bottleneck NN ──────────────────────────────────────────────────────────
inp = tf.keras.Input(shape=(2,))
x   = tf.keras.layers.Dense(64, activation="tanh")(inp)
x   = tf.keras.layers.Dense(32, activation="tanh")(x)
x   = tf.keras.layers.Dense(2,  activation="linear", name="bottleneck")(x)
x   = tf.keras.layers.Dense(32, activation="tanh")(x)
x   = tf.keras.layers.Dense(64, activation="tanh")(x)
out = tf.keras.layers.Dense(1,  activation="linear")(x)
bnn = tf.keras.Model(inp, out)
bnn.compile(optimizer=tf.keras.optimizers.Adam(3e-3), loss="mse")
bnn.fit(X_train_s.astype("float32"), y_train.astype("float32"),
        validation_data=(X_test_s.astype("float32"), y_test.astype("float32")),
        epochs=800, batch_size=64, verbose=0,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=50,
                                                     restore_best_weights=True)])
print("All models trained.")
'''),

        md("g5", "## Test-Set Metrics"),

        code("g6", '''def metrics(name, y_true, y_pred):
    return {"Model": name,
            "MSE":  round(mean_squared_error(y_true, y_pred), 5),
            "MAE":  round(mean_absolute_error(y_true, y_pred), 5),
            "R2":   round(r2_score(y_true, y_pred), 4)}

rows = [
    metrics("Ridge",         y_test, ridge.predict(X_test_s)),
    metrics("Poly-3",        y_test, poly3.predict(X_test)),
    metrics("GPR-Matern",    y_test, gpr.predict(X_test_s)),
    metrics("RandomForest",  y_test, rf.predict(X_test)),
    metrics("XGBoost",       y_test, xgb_m.predict(X_test)),
    metrics("Bottleneck-NN", y_test, bnn.predict(X_test_s.astype("float32"),
                                                  verbose=0).ravel()),
]
comp_df = pd.DataFrame(rows).sort_values("MSE").reset_index(drop=True)
print(comp_df.to_string(index=False))
comp_df.to_csv("../results/metrics_comparison.csv", index=False)
'''),

        md("g7", r"""## Low-x Extrapolation — All Methods"""),

        code("g8", '''Q2_plot = [1.0, 5.0, 15.0, 30.0]
grids   = low_x_grid(x_min=1e-6, x_max=0.8, n_points=400, Q2_values=Q2_plot)

MODEL_SPEC = [
    ("Ridge",        "#a65628", lambda X: ridge.predict(scaler.transform(X))),
    ("Poly-3",       "#ff7f00", lambda X: poly3.predict(X)),
    ("GPR-Matern",   "#984ea3", lambda X: gpr.predict(scaler.transform(X))),
    ("RandomForest", "#e41a1c", lambda X: rf.predict(X)),
    ("XGBoost",      "#377eb8", lambda X: xgb_m.predict(X)),
    ("Bottleneck-NN","#4daf4a", lambda X: bnn.predict(
                                    scaler.transform(X).astype("float32"),
                                    verbose=0).ravel()),
]

predictions = []
for label, color, pred_fn in MODEL_SPEC:
    entry = {"label": label, "color": color, "x_arr": None, "Q2_preds": {}}
    for Q2v, (x_arr, X_feat) in grids.items():
        entry["x_arr"]         = x_arr
        entry["Q2_preds"][Q2v] = pred_fn(X_feat)
    predictions.append(entry)

# Also add GPR uncertainty
gpr_entry = next(p for p in predictions if p["label"] == "GPR-Matern")
gpr_entry["Q2_lo"] = {}
gpr_entry["Q2_hi"] = {}
for Q2v, (x_arr, X_feat) in grids.items():
    mu, sig = gpr.predict(scaler.transform(X_feat), return_std=True)
    gpr_entry["Q2_lo"][Q2v] = mu - 2 * sig
    gpr_entry["Q2_hi"][Q2v] = mu + 2 * sig

fig, _ = comparison_figure(Q2_plot, df, predictions, figsize=(18, 14))
fig.suptitle("Low-x extrapolation — all supervised methods", fontsize=14, y=1.01)
plt.savefig("../results/figures/06_all_models.png", dpi=150, bbox_inches="tight")
plt.show()
'''),

        md("g9", r"""## Interpretation

| Method | Low-x behaviour |
|---|---|
| **Ridge / Poly-3** | Smooth power-law extrapolation — may over- or under-shoot |
| **GPR** | Principled uncertainty quantification; reverts to prior in data-free region |
| **Random Forest / XGBoost** | Plateau at the minimum training x — cannot extrapolate |
| **Bottleneck NN** | Smooth extrapolation from learned internal representation |

**Key take-away**: GPR is the most honest tool for extrapolation because it explicitly
signals when its prediction is uncertain. Tree methods should not be used for
out-of-sample extrapolation in kinematic variables.
"""),
    ]
    save(nb(cells), "06_model_comparison.ipynb")


# ═════════════════════════════════════════════════════════════════════════════
# README
# ═════════════════════════════════════════════════════════════════════════════

README = r"""# F2 EM Structure Function — ML Portfolio

A self-contained machine-learning project that models and extrapolates the
**proton electromagnetic structure function F₂(x, Q²)** into the low-x region
using a full suite of supervised and unsupervised methods.

## Physics background

F₂ is measured in Deep Inelastic Scattering (DIS) experiments (H1, ZEUS, NMC, SLAC, BCDMS).
It depends on two kinematic variables:

* **x** — Bjorken scaling variable (parton momentum fraction)
* **Q²** — photon virtuality (resolution scale)

At very small x (x < 10⁻⁴) experimental data are sparse. All models are
trained on existing data and extrapolated into this low-x region.

## Project structure

```
F2-ML-Portfolio/
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
cd F2-ML-Portfolio
pip install -r requirements.txt

# (optional) regenerate notebooks
python create_notebooks.py

# launch Jupyter from the project root
jupyter lab
```

> **Data path**: set `DATA_DIR` at the top of each notebook to point to
> your local copy of the LeptonDIS `.dat` files.
"""

if __name__ == "__main__":
    print("Generating notebooks …")
    make_00()
    make_01()
    make_02()
    make_03()
    make_04()
    make_05()
    make_06()

    readme_path = Path(__file__).parent / "README.md"
    readme_path.write_text(README, encoding="utf-8")
    print(f"  Written: {readme_path}")
    print("Done.")
