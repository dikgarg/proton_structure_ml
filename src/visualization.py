"""
Shared plotting utilities for the F2 ML portfolio.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Consistent colour per experiment
EXP_COLORS = {
    "H1":    "#e41a1c",
    "ZEUS":  "#377eb8",
    "NMC92": "#4daf4a",
    "NMC97": "#984ea3",
    "SLAC":  "#ff7f00",
    "BCDMS": "#a65628",
}


# ── Data plots ────────────────────────────────────────────────────────────────

def plot_coverage(df, ax=None, title="Data coverage in the $(x, Q^2)$ plane"):
    """Scatter of every (x, Q²) point, coloured by experiment."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))
    for exp, grp in df.groupby("experiment"):
        ax.scatter(
            grp["x"], grp["Q2"],
            s=12, alpha=0.7, label=exp,
            color=EXP_COLORS.get(exp, "grey"),
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$x$", fontsize=12)
    ax.set_ylabel(r"$Q^2\;[\mathrm{GeV}^2]$", fontsize=12)
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, which="both", ls="--", alpha=0.3)
    return ax


def plot_F2_vs_x(df, Q2_centres, delta, figsize=None, ncols=3): #tolerance bins are not done correctly (i.e. delta)!!!! FIX
    """Grid of F₂ vs x panels, one per Q² bin.

    delta_log=0.05 means Q2 is within +/-10% of the centre value.
    A narrow window is needed because experiments measure at different Q2 points.
    """
    delta_log = np.log10(delta)
    nrows = int(np.ceil(len(Q2_centres) / ncols))
    if figsize is None:
        figsize = (5 * ncols, 4 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    axes_flat = axes.ravel()

    for i, Q2c in enumerate(Q2_centres):
        ax = axes_flat[i]
        lo = 10 ** (np.log10(Q2c) - delta_log)
        hi = 10 ** (np.log10(Q2c) + delta_log)
        sub = df[(df["Q2"] >= lo) & (df["Q2"] <= hi)] 
        for exp, grp in sub.groupby("experiment"):
            ax.errorbar(
                grp["x"], grp["F2"],
                yerr=[grp["sigma_dn"], grp["sigma_up"]],
                fmt="o", ms=3, alpha=0.8,
                color=EXP_COLORS.get(exp, "grey"),
                label=exp,
            )
        ax.set_xscale("log")
        ax.set_xlabel(r"$x$", fontsize=10)
        ax.set_ylabel(r"$F_2^p$", fontsize=10)
        ax.set_title(rf"$Q^2 = {Q2c} \pm 5\%\;\mathrm{{GeV}}^2$", fontsize=10)
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, ls="--", alpha=0.3)

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)
    fig.tight_layout()
    return fig, axes


# ── Model overlay ─────────────────────────────────────────────────────────────

def overlay_predictions(ax, x_arr, y_pred, label, color, lw=2.0, ls="-",
                         y_lo=None, y_hi=None):
    """Add one model's prediction curve (and optional uncertainty band) to *ax*."""
    ax.plot(x_arr, y_pred, color=color, lw=lw, ls=ls, label=label)
    if y_lo is not None and y_hi is not None:
        ax.fill_between(x_arr, y_lo, y_hi, color=color, alpha=0.15)
    return ax


def comparison_figure(Q2_values, data_df, predictions, figsize=None):
    """Panel figure comparing multiple model predictions against data.

    Parameters
    ----------
    Q2_values  : list of exact Q² values for the panels
    data_df    : full DataFrame with experimental points
    predictions: list of dicts, each with keys
                   'label', 'color', 'x_arr',
                   'Q2_preds'  : dict { Q2_value -> y_pred array }
                   'Q2_lo'     : dict { Q2_value -> lower band }  (optional)
                   'Q2_hi'     : dict { Q2_value -> upper band }  (optional)
    """
    ncols = min(len(Q2_values), 3)
    nrows = int(np.ceil(len(Q2_values) / ncols))
    if figsize is None:
        figsize = (5 * ncols, 4 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    axes_flat = axes.ravel()

    for i, Q2c in enumerate(Q2_values):
        ax = axes_flat[i]
        sub = data_df[data_df["Q2"] == Q2c]
        for exp, grp in sub.groupby("experiment"):
            ax.errorbar(
                grp["x"], grp["F2"],
                yerr=[grp["sigma_dn"], grp["sigma_up"]],
                fmt="o", ms=3, alpha=0.7,
                color=EXP_COLORS.get(exp, "grey"),
                label=exp,
            )
        for pred in predictions:
            y = pred["Q2_preds"].get(Q2c)
            if y is None:
                continue
            overlay_predictions(
                ax, pred["x_arr"], y,
                label=pred["label"], color=pred["color"],
                y_lo=pred.get("Q2_lo", {}).get(Q2c),
                y_hi=pred.get("Q2_hi", {}).get(Q2c),
            )
        ax.set_xscale("log")
        ax.set_xlabel(r"$x$", fontsize=10)
        ax.set_ylabel(r"$F_2^p$", fontsize=10)
        ax.set_title(rf"$Q^2 = {Q2c} \pm 5\%\;\mathrm{{GeV}}^2$", fontsize=10)
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, ls="--", alpha=0.3)

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)
    fig.tight_layout()
    return fig, axes
