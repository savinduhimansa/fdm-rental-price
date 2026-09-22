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

# Step 5 validity rules
RENT_FLOOR = 250                  # 5.9: $100-$249 are mostly weekly/nightly rates, fees or sales,
                                  #      so the effective lower price limit becomes $250
SQFT_MIN, SQFT_MAX = 100, 10_000  # 5.1: <100 are typos/not entered; >10,000 are placeholders/typos
BEDS_MAX = 8                      # 5.2: only 1,000 and 1,100 bedrooms exist above 8
BATHS_MIN, BATHS_MAX = 0.5, 8     # 5.3: 0 baths means "not entered" (ordinary apartments)
MIN_SQFT_PER_ROOM = 70            # 5.4: sqfeet / (beds + 1) below this is physically impossible
MAX_EXTRA_BATHS = 3               # 5.4: baths > beds + 3 were all errors or per-room pricing
US_LAT_RANGE = (18, 72)           # 5.5: box around U.S. territory (incl. Alaska, Hawaii)
US_LON_RANGE = (-180, -65)
FAR_KM = 500                      # 5.6: pin this far from its region centre -> location conflict

# Step 6 rules
PPSF_MIN, PPSF_MAX = 0.25, 10     # 6.5: price per sq ft outside this is a room/nightly/sale price
                                  #      or a size typo (e.g. a 100 sq ft studio described as 475)
EXCLUDED_TYPES = ["land", "assisted living"]   # 6.6: RV lots, fish camps, adult foster care

# ---------------------------------------------------------------------------
# Columns
# ---------------------------------------------------------------------------
TARGET = "price"
BINARY_COLS = ["cats_allowed", "dogs_allowed", "smoking_allowed",
               "wheelchair_access", "electric_vehicle_charge", "comes_furnished"]
MODEL_FEATURES = ["region_url", "state", "type", "sqfeet", "beds", "baths", *BINARY_COLS,
                  "laundry_options", "parking_options", "lat", "long"]

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
STEP5_FILE = PROCESSED_DIR / "housing_step5.parquet"   # after Steps 2-5 (input to Step 6 EDA)
CLEAN_FILE = PROCESSED_DIR / "housing_clean.parquet"   # after all cleaning rules (Steps 2-6)
TRAIN_FILE = PROCESSED_DIR / "train.parquet"           # Step 7: 80%, used for all analysis/training
TEST_FILE = PROCESSED_DIR / "test.parquet"             # Step 7: 20%, opened ONCE for final evaluation
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


def load_train():
    """The training set from Step 7. All EDA from Step 8 onward and all model training use this."""
    return pd.read_parquet(TRAIN_FILE)


def load_test():
    """The locked test set from Step 7.

    Do NOT use this for EDA, feature design or model selection. Open it once, at the very
    end, to evaluate the final chosen model. Every earlier look makes the score optimistic."""
    return pd.read_parquet(TEST_FILE)


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
    log = log[log["step"].astype(str).str.split("-").str[0] != str(step)]
    log.to_csv(OBS_FILE, index=False)


def show_obs(step=None):
    """Show the log. Pass a step prefix (e.g. "2") to see only that step."""
    log = _read_obs()
    if step is not None:
        log = log[log["step"].astype(str).str.split("-").str[0] == str(step)]
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

# Step 8
RARE_MIN_SHARE = 0.01   # categories with under 1% of training listings are grouped as "other"

# Step 9
RARE_MIN_COUNT = 50   # 9.4: the rare categories differ in rent (type 32%, parking 65% spread), so every
                      #      category with 50+ training listings keeps its own one-hot column

# Step 11 (error-analysis groups, also used when the models are evaluated)
PRICE_BANDS = [250, 750, 1_000, 1_500, 2_500, 4_000, 10_000]   # band edges for monthly rent (USD)
REGION_SIZE_BANDS = [0, 30, 100, 500, float("inf")]            # training listings per region
STUDENT_LAYOUT_MIN_BEDS = 4   # 10.7: 4+ bedrooms with a bathroom per bedroom are often priced per bedroom


def price_band(price):
    """Label each rent (or predicted rent) with its price band, for error analysis by band."""
    labels = [f"${lo:,}-${hi:,}" for lo, hi in zip(PRICE_BANDS[:-1], PRICE_BANDS[1:])]
    return pd.cut(price, bins=PRICE_BANDS, labels=labels, include_lowest=True)


def region_size_band(regions, train_counts):
    """Label each listing by how many TRAINING listings its region has (train_counts = value_counts)."""
    edges = REGION_SIZE_BANDS
    labels = [f"{int(lo)}+" if hi == float("inf") else f"{int(lo)}-{int(hi) - 1}"
              for lo, hi in zip(edges[:-1], edges[1:])]
    return pd.cut(regions.map(train_counts).fillna(0), bins=edges, labels=labels, right=False)


def student_layout(df):
    """True for homes with 4+ bedrooms and at least one bathroom per bedroom (Step 10.7).
    Uses only beds and baths, so it is available at prediction time."""
    return df["beds"].ge(STUDENT_LAYOUT_MIN_BEDS) & df["baths"].ge(df["beds"])
