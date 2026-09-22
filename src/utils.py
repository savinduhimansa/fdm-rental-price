"""
Shared helpers for the FDM rental-price project.
Import this in every notebook so paths, settings and logging stay identical.

    from src.utils import *
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Confirmed cleaning rules (decided in the EDA notebooks)
# ---------------------------------------------------------------------------
PRICE_MIN = 100      # Step 2: below this, prices are placeholders ($0, $1) or errors
PRICE_MAX = 10_000   # Step 2: above this, mostly sale prices and nonsense values;
                     # the system is scoped to rents up to $10,000

# ---------------------------------------------------------------------------
# Paths
# utils.py lives in <project>/src/, so the project root is one level up.
# This works no matter which folder the notebook is opened from.
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
INTERIM_DIR = ROOT / "data" / "interim"
PROCESSED_DIR = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "reports"
FIG_DIR = REPORT_DIR / "figures"
MODEL_DIR = ROOT / "models"

RAW_FILE = RAW_DIR / "housing.csv"
RAW_PARQUET = INTERIM_DIR / "housing_raw.parquet"
OBS_FILE = REPORT_DIR / "observations_log.csv"

for _d in [INTERIM_DIR, PROCESSED_DIR, FIG_DIR, MODEL_DIR]:
    _d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Display and plot settings
# ---------------------------------------------------------------------------
def setup_display():
    """Apply the same pandas display and plot style in every notebook."""
    np.random.seed(RANDOM_STATE)
    pd.set_option("display.max_columns", 50)
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", "{:,.2f}".format)
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams["figure.figsize"] = (10, 5)
    plt.rcParams["figure.dpi"] = 100


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_raw():
    """Load the untouched raw data. Uses the fast parquet copy if it exists."""
    if RAW_PARQUET.exists():
        return pd.read_parquet(RAW_PARQUET)
    df = pd.read_csv(RAW_FILE, low_memory=False)
    df.to_parquet(RAW_PARQUET, index=False)
    return df


# ---------------------------------------------------------------------------
# Observations log (Step 13 table)
# The log lives in a CSV file, not in notebook memory, so every notebook
# adds to the same table and nothing is lost between sessions.
# ---------------------------------------------------------------------------
_OBS_COLS = ["step", "observation", "evidence", "decision"]


def _read_obs():
    if OBS_FILE.exists():
        return pd.read_csv(OBS_FILE)
    return pd.DataFrame(columns=_OBS_COLS)


def log_obs(step, observation, evidence, decision):
    """Append one EDA finding and its decision to reports/observations_log.csv.
    Re-running a cell does not create duplicate entries."""
    new = pd.DataFrame([{"step": step, "observation": observation,
                         "evidence": evidence, "decision": decision}])
    log = pd.concat([_read_obs(), new], ignore_index=True)
    log = log.drop_duplicates(subset=["step", "observation"], keep="last")
    log.to_csv(OBS_FILE, index=False)


def clear_obs(step):
    """Remove all log entries whose step starts with `step` (e.g. "2").
    Call this at the top of a step's logging cell, so re-running it with
    changed numbers replaces the old entries instead of adding new ones."""
    log = _read_obs()
    log = log[~log["step"].astype(str).str.startswith(str(step))]
    log.to_csv(OBS_FILE, index=False)


def show_obs(step=None):
    """Show the log. Pass a step prefix (e.g. "2") to see only that step."""
    log = _read_obs()
    if step is not None:
        log = log[log["step"].astype(str).str.startswith(str(step))]
    return log.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def save_fig(name):
    """Save the current figure to reports/figures/<name>.png for the report."""
    plt.savefig(FIG_DIR / f"{name}.png", dpi=150, bbox_inches="tight")


def plot_sample(df, n=50_000):
    """Fixed random sample for charts only. Statistics always use the full data."""
    return df.sample(n=min(n, len(df)), random_state=RANDOM_STATE)
