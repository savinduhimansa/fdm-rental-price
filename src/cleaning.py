"""
Data cleaning for the rental-price project.

Every rule here was decided, with evidence, in the EDA notebooks (Steps 2-6).
The notebooks, the training code and the report all use this one function,
so the rules can never drift apart.

    from src.cleaning import clean_data
    df_clean, funnel = clean_data(df_raw)

None of these rules learns anything from prices, so applying them before the
train/test split does not cause leakage.
"""
import numpy as np
import pandas as pd

from src.utils import (
    RENT_FLOOR, PRICE_MAX, SQFT_MIN, SQFT_MAX, BEDS_MAX, BATHS_MIN, BATHS_MAX,
    MIN_SQFT_PER_ROOM, MAX_EXTRA_BATHS, US_LAT_RANGE, US_LON_RANGE, FAR_KM,
    PPSF_MIN, PPSF_MAX, EXCLUDED_TYPES, MODEL_FEATURES, TARGET,
)


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km between points given in degrees."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


def in_us_box(lat, lon):
    """True where the coordinates fall inside the U.S. bounding box."""
    return lat.between(*US_LAT_RANGE) & lon.between(*US_LON_RANGE)


def region_centres(df):
    """Median lat/long of each Craigslist site, using only coordinates inside the U.S."""
    ok = in_us_box(df["lat"], df["long"])
    return df[ok].groupby("region_url")[["lat", "long"]].median()


def distance_to_region_centre(df, centres=None):
    """Distance (km) from each listing's pin to its region's centre; NaN if no pin."""
    centres = region_centres(df) if centres is None else centres
    c_lat = df["region_url"].map(centres["lat"])
    c_lon = df["region_url"].map(centres["long"])
    return haversine_km(df["lat"], df["long"], c_lat, c_lon)


def invalid_core_facts(df):
    """True where size, bedrooms or bathrooms are impossible or inconsistent (Step 5.1-5.4)."""
    return (
        ~df["sqfeet"].between(SQFT_MIN, SQFT_MAX)
        | df["beds"].gt(BEDS_MAX)
        | ~df["baths"].between(BATHS_MIN, BATHS_MAX)
        | (df["sqfeet"] / (df["beds"] + 1)).lt(MIN_SQFT_PER_ROOM)
        | (df["baths"] > df["beds"] + MAX_EXTRA_BATHS)
    )


def clean_data(df, through_step=6, verbose=True):
    """Apply every confirmed cleaning rule, in the order decided in Step 4.

    Parameters
    ----------
    df           : raw listings
    through_step : 6 (default) applies every rule. 5 skips the Step 6 rules; it is only
                   used by the Step 6 notebook, which needs to see the data before them.

    Returns
    -------
    df_clean : DataFrame, one row per distinct listing
    funnel   : DataFrame with the number of rows left after each stage
    """
    stages = [("1. Raw data", len(df))]
    out = df.copy()

    # Step 2 + 5.9: plausible monthly rent
    out = out[out[TARGET].between(RENT_FLOOR, PRICE_MAX)]
    stages.append((f"2. Price ${RENT_FLOOR}-${PRICE_MAX:,} (Steps 2, 5.9)", len(out)))

    # Step 5.1-5.4: impossible size / bedrooms / bathrooms -> remove the row
    out = out[~invalid_core_facts(out)]
    stages.append(("3. Valid size, beds, baths (Steps 5.1-5.4)", len(out)))

    # Step 5.5: pins outside the U.S. -> blank (region still known; filled later in the pipeline)
    outside = out["lat"].notna() & ~in_us_box(out["lat"], out["long"])
    out.loc[outside, ["lat", "long"]] = np.nan

    # Step 5.6: pins far from their own region -> location conflict -> remove the row
    far = distance_to_region_centre(out) > FAR_KM   # NaN distance (no pin) compares as False
    out = out[~far]
    stages.append(("4. Location consistent (Steps 5.5-5.6)", len(out)))

    # Step 6.5-6.6: implausible price per sq ft, and property types that are not homes
    if through_step >= 6:
        ppsf = out[TARGET] / out["sqfeet"]
        out = out[ppsf.between(PPSF_MIN, PPSF_MAX) & ~out["type"].isin(EXCLUDED_TYPES)]
        stages.append((f"5. Price per sq ft ${PPSF_MIN}-${PPSF_MAX:g}, homes only (Step 6)", len(out)))

    # Step 5.7: one state label per Craigslist site (its most common label)
    site_state = out.groupby("region_url")["state"].agg(lambda s: s.mode().iloc[0])
    out["state"] = out["region_url"].map(site_state)

    # Step 4: one row per group of listings that are identical to the model
    out = out.drop_duplicates(subset=MODEL_FEATURES + [TARGET])
    stages.append((f"{len(stages) + 1}. Deduplicated (Step 4)", len(out)))

    funnel = pd.DataFrame(stages, columns=["stage", "rows"]).set_index("stage")
    funnel["removed"] = (-funnel["rows"].diff()).fillna(0).astype(int)
    funnel["%_of_raw_remaining"] = funnel["rows"] / len(df) * 100

    if verbose:
        print(f"clean_data: {len(df):,} -> {len(out):,} rows "
              f"({len(out) / len(df) * 100:.1f}% kept); "
              f"{int(outside.sum()):,} pins outside the U.S. blanked")

    return out.reset_index(drop=True), funnel
