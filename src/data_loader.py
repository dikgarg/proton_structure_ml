"""
Author: D. Garg
May 10, 2026

Data loading utilities for the F2 EM structure function ML project.

File format (LeptonDIS .dat files):
    Whitespace-separated, comment lines start with '#'.
    Columns: x  Q2  F2  tot+  tot-  stat+  stat-  sys+  sys-  sys2+  sys2-  q0
"""

import glob
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

_COLS = [
    "x", "Q2", "F2",
    "tot_up", "tot_dn", "stat_up", "stat_dn",
    "sys_up", "sys_dn", "sys2_up", "sys2_dn", "q0",
]


def load_single_file(filepath):
    """Read one LeptonDIS .dat file and return a cleaned DataFrame.

    Adds:
      - 'experiment' column (first token of the filename stem, e.g. 'H1')
      - 'sigma' column (symmetric total uncertainty)
    Drops rows with non-positive x, Q2, or F2 (needed for log transforms).
    """
    df = pd.read_csv(
        filepath,
        sep=r"\s+",
        header=None,
        comment="#",
        names=_COLS,
    )
    stem = Path(filepath).stem          # e.g. 'H1-F2p-q0'
    df["experiment"] = stem.split("-")[0]
    df["sigma_up"] = df["tot_up"].abs()
    df["sigma_dn"] = df["tot_dn"].abs()
    df = df[(df["x"] > 0) & (df["Q2"] > 0) & (df["F2"] > 0)].copy()
    return df


def load_lepton_dis(data_dir, pattern="*-F2p*-q0.dat"):
    """Load all matching experiment files from *data_dir* into one DataFrame."""
    files = sorted(glob.glob(str(Path(data_dir) / pattern)))
    if not files:
        raise FileNotFoundError(
            f"No files matching '{pattern}' in {data_dir}"
        )
    frames = [load_single_file(f) for f in files]
    combined = pd.concat(frames, ignore_index=True)
    print(f"Loaded {len(files)} experiment files → {len(combined)} data points")
    for f in files:
        print(f"  {Path(f).name}")
    return combined


def add_log_features(df):
    """Return a copy of *df* with log10(x) and log10(Q2) columns added."""
    out = df.copy()
    out["log10_x"] = np.log10(out["x"])
    out["log10_Q2"] = np.log10(out["Q2"])
    return out


def split_data(df, test_size=0.2, seed=42):
    """Randomly split *df* into train / test DataFrames."""
    train, test = train_test_split(
        df, test_size=test_size, random_state=seed, shuffle=True #shuffles the order of data b4 splitting
    )
    return train.reset_index(drop=True), test.reset_index(drop=True) #drops the original indexing of rows


def get_Xy(df, features=("log10_x", "log10_Q2"), target="F2"):
    """Extract feature matrix X and target array y as float64 numpy arrays."""
    X = df[list(features)].to_numpy(dtype=np.float64)
    y = df[target].to_numpy(dtype=np.float64)
    return X, y


def low_x_grid(x_min=1e-6, x_max=1e-2, n_points=300, Q2_values=(1.0, 5.0, 15.0, 30.0)):
    """Build prediction grids for low-x extrapolation.

    Returns a dict: Q2_value → (x_array, X_feature_matrix)
    where X_feature_matrix columns are [log10_x, log10_Q2].
    """
    x_arr = np.logspace(np.log10(x_min), np.log10(x_max), n_points)
    grids = {}
    for Q2 in Q2_values:
        lx = np.log10(x_arr)
        lQ2 = np.full_like(lx, np.log10(Q2)) #same log10(Q2) value with len(x_arr)
        grids[float(Q2)] = (x_arr, np.column_stack([lx, lQ2]))
    return grids
