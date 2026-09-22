# EDA Findings – Rental Price Estimation System

This document records the **interpretation** of each EDA step: what the results mean and what was decided. The notebooks hold the code and the raw outputs and are not edited after they are run. The observations log (`reports/observations_log.csv`) holds the one-line version of every finding.

Steps 1–6 are summarised at the end of their own notebooks (`01_eda.ipynb` to `05_eda_outliers.ipynb`). This document continues from Step 7.

---

## Step 7 – The train / test split (`06_split.ipynb`)

| Finding | Evidence | Decision |
|---|---|---|
| **37.1%** of clean listings have a close relative: the same features reposted at a different price, or the same description with a small edit | 7.3 | Treat each "family" of relatives as one group |
| A normal random stratified split would put **34.7%** of test listings next to a relative in the training set | 7.4 | Use a grouped split (`StratifiedGroupKFold`), which brings this to 0% |
| Training set: **150,404** listings (80.0%); test set: **37,602** | 7.5 | Test set locked away until the final evaluation |
| All four leakage checks pass: no shared id, no identical listing, no identical features at another price, no shared long description | 7.7 | The test set is genuinely unseen |

**Why this matters:** Step 4 removed exact copies, but more than a third of the remaining listings still had a near-copy. Had we split randomly, about a third of the "final exam" would have been listings whose near-twin the model had already studied, and the test score would have been clearly over-optimistic. The grouped split is what makes the final score trustworthy. The training set keeps a `group_id` column so that cross-validation during model selection can be grouped in the same way (`GroupKFold`).

---

## Step 8 – Univariate distributions (`07_eda_univariate.ipynb`, training set)

### Numeric features

| Finding | Evidence | Decision |
|---|---|---|
| `sqfeet` is strongly right-skewed (skew **2.53**, kurtosis 16.4); after `log1p` it is almost symmetric (skew **0.14**) | 8.2 | Log-transform and standardise for Linear Regression, k-NN and the MLP; raw values for tree models |
| **21.4%** of sizes are exact multiples of 100 sq ft (about 1% expected by chance) and **30.5%** multiples of 50. The most common sizes are 1,000, 900 and 1,200 | 8.2 (cont.) | No action. Report as a **data limitation**: many sizes are estimates, which limits how precisely size can predict rent |
| `beds` (skew 0.56) and `baths` (skew 0.99) are short-tailed counts; log makes `beds` left-skewed (−0.59) | 8.3 | Keep as numeric counts, standardised, no log |
| Only **0.12%** of listings have 6 or more bedrooms; 11.5% have a half-bathroom | 8.3 | Predictions for very large homes rest on few examples (imbalance, Step 11) |
| **969** training listings (0.64%) have no coordinates | 8.4 | Fill from the region centre, learned on training data only |
| Coordinates are multi-peaked. Coverage is densest on the coasts and around big metros, sparse in the Mountain West and Great Plains; Alaska and Hawaii form separate clusters | 8.4 map | Coordinates mainly serve the tree models; the linear model receives location through the region encoding |

### Categorical features

| Finding | Evidence | Decision |
|---|---|---|
| `region_url` has **413** values: median 200 listings per region, but **29 regions have fewer than 30** (the smallest, Big Bend, has 4) and 131 have fewer than 100 | 8.5 | **Target-encode** with smoothing (small regions pulled towards the overall average) and cross-fitting (scikit-learn `TargetEncoder`) |
| `state` has 51 values, each with at least 125 listings; the top 15 states hold 61.4% of listings | 8.5 | One-hot encode. (The 8.1 overview flags `state` as "high-cardinality" only because its 51 values are just above that table's cut-off of 50; 51 one-hot columns are no problem.) |
| `type`: four rare types under 1%: cottage/cabin (0.34%), flat (0.25%), loft (0.24%), in-law (0.09%), together 0.92% | 8.6 | Group as "other", **if** Step 9 shows their rents are similar |
| `parking_options`: two rare values: **no parking (0.73%)** and **valet parking (0.07%)** | 8.6 | **Caution:** these are opposites. Blind grouping would merge the cheapest and most expensive parking situations. Step 9 checks their rents before deciding |
| `laundry_options`: no rare categories; 18.1% missing. `parking_options`: 31.5% missing | 8.1, 8.6 | One-hot encode, with `"missing"` as its own category (Step 3) |

### Binary flags

| Flag | "Yes" share | Interpretation |
|---|---|---|
| `electric_vehicle_charge` | 1.9% | Rare; the only near-constant feature (98.1% "no") |
| `comes_furnished` | 5.6% | Uncommon |
| `wheelchair_access` | 8.3% | Uncommon |
| `dogs_allowed` | 66.5% | Balanced |
| `smoking_allowed` | 67.4% | Suspiciously high for U.S. rentals; possibly a Craigslist default. **Reliability limitation** |
| `cats_allowed` | 68.7% | Balanced |

**Decision:** all flags are used as 0/1 and standardised for the sensitive models. Step 9 checks which ones relate to rent.

### Summary of Step 8

The training set confirms the Step 6 transform decisions (log for size, plain counts for beds and baths). The main new findings are about **encoding**. With 413 regions, many of them small, location needs target encoding with smoothing rather than one-hot columns. Rare categories are few and small, but in `parking_options` they are *opposites*, so the standard "group everything under 1%" rule may not suit every column; Step 9 decides that from their rents. Two data limitations came out of this step: **rounded sizes** (heaping) and the unreliable **`smoking_allowed`** flag.





## Step 9 – Relationships with price (training set, `08_eda_relationships.ipynb`)

**Location is the strongest driver of rent.** On its own, `region_url` reproduces 48.1% of the variation in log rent (η²), `state` 31.1% and longitude 22.4%. State medians range from $700 (Missouri) to $1,999 (Hawaii), 2.9×; region medians (100+ listings) from $622 (Carbondale, IL) to $2,900 (Santa Barbara), 4.7×. New York State sits near the national median ($1,190) because it mixes New York City ($2,400) with cheap upstate regions, so state is too coarse and the platform must ask for the region. Rents differ mainly east–west (longitude η² 22.4% vs latitude 7.1%): expensive coasts, cheaper interior.

**Size matters moderately.** Pearson (log–log) 0.381, Spearman 0.360; elasticity 0.45 (a 10% larger home rents for about 4.5% more). Binned η² (14.4%) equals the squared correlation (14.5%), so the log–log relationship is close to linear. Bathrooms (η² 10.8%) relate to rent more steadily than bedrooms (4.6%).

**Amenities follow a clear ladder.** Laundry: no laundry on site $850 → w/d in unit $1,372 (+19.3% vs overall median). Parking: attached garage +36.4%, valet +64.7%. Flags: EV charging +47.2%, wheelchair access +14.0% (proxies for newer buildings), smoking allowed −14.5%, pets ≈ 0%. Blank laundry/parking sit mid-range, confirming the `"missing"` category from Step 3.

**Results that look odd one-at-a-time** (studios ≈ 1-bedrooms, "no parking" dearer than "off-street", a low size elasticity) are probably location effects; Step 10.4 tests them within regions. Two anomalies (5 beds cheaper than 4; a very wide 4-bath box) suggest per-bedroom pricing in student housing; Step 10.7 tests this.

**Rare categories must not be merged.** Median-rent spread among rare categories: type 32%, parking 65% (both above the 20% limit). Decision: every category with 50+ training listings keeps its own one-hot column (`RARE_MIN_COUNT = 50`, `OneHotEncoder(min_frequency=..., handle_unknown="infrequent_if_exist")`).

**No funnel in the region chart.** Small regions (< 30 listings) are mostly cheaper than the national median and less spread out (std of log medians 0.16) than large regions (0.27). Region size is itself linked to rent, and real market differences outweigh small-sample noise. Smoothed target encoding is kept for the tiniest regions; Step 10.5 tests whether a state-level fallback beats the national one.

**Features under 1% η² on their own:** wheelchair access, furnished, cats, dogs. Kept for now; re-checked with mutual information in Step 10.3.
