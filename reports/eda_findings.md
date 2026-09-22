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




## Step 10 – Multivariate relationships (training set, `09_eda_multivariate.ipynb`)

**Redundancy.** The size features overlap (Spearman: sqfeet–beds 0.80, sqfeet–baths 0.72, beds–baths 0.66), and so do the pet flags (cats–dogs 0.85). No VIF reaches the moderate level of 5 (highest 3.72, cats_allowed; size trio 3.07 / 2.81 / 2.08). But the bedroom coefficient flips sign once size and baths are held constant (+9.6% alone → −11.7% together per standard deviation): at the same floor space, more bedrooms means smaller rooms. Ridge is used for the linear model, and `beds` is interpreted only "at the same size".

**Feature ranking confirmed.** Mutual information (corrected with a shuffled-target baseline, 40,000-listing sample) and η² agree (rank agreement 0.81): location first (region, longitude, latitude, state), then size. Latitude rises from #8 (η²) to #3 (MI), showing that coordinates carry more information together. MI for weak flags is probably inflated by repeated exact prices within listing families. No feature is dropped; weak features will be judged by permutation importance in grouped CV.

**Location confounding.** Region median size and rent are unrelated overall (ρ = +0.08); only the most expensive metros have small homes. The size elasticity changes little within regions (0.45 → 0.49), but within regions size explains 32.2% of the remaining variation vs 14.5% nationally. Location explains the Step 9 puzzles: studios vs 1-bed +0.6% observed → −11.1% within region; the 5-bedroom dip disappears; "no parking" vs "off-street" +15.0% → +4.8%.

**Region vs state.** Variance of log rent: 31.1% between states, 17.0% between regions within states, 51.9% within regions. For small regions (< 30 listings), the state average (excluding the region) is twice as close to the region's rent as the national average (typical error 22.7% vs 44.0%; closer for 83% of them); small regions are on average 29% cheaper than the national level. Decision: keep region (target-encoded) and state (one-hot), and test a state-prior region encoding against the default `TargetEncoder`.

**Interactions are modest.** Within-region size elasticity ranges 0.40 (PA) – 0.53 (CA) with no trend by rent level (ρ = −0.04); the order of property types by $/sq ft is stable across the five largest states (agreement 0.91). California costs about twice as much per sq ft for every type, a multiplicative effect that the log target turns into an additive one. Ridge is expected to be a competitive baseline.

**Per-bedroom pricing (student housing).** 4-bed homes with a bath per bedroom: median $830 vs $1,500 for other 4-beds, $0.53 vs $0.84 per sq ft (−37%; 5 beds −34%); per-bedroom wording 7.6% vs 1.9% of descriptions; the 876 suspects (0.58%) concentrate in university towns (Gainesville, San Marcos, Tallahassee, Tippecanoe, Champaign-Urbana, College Station…). Recorded as a data limitation (cleaning rules were frozen before the split); tracked as an error-analysis group; optional indicator feature (4+ beds, baths ≥ beds).




## Step 11 – Imbalance, regression interpretation (training set, `10_eda_imbalance.ipynb`)

**Target imbalance.** 95.0% of training listings rent for $2,500 or less (73.9% for $1,500 or less); the $4,000–$10,000 band holds only 0.7% (1,104 listings), 47× fewer than the busiest band ($1,000–$1,500, 34.5%). The log target reduces skew from 2.70 to 0.39 but does not remove the thin tail. Decision: no oversampling or weighting; model `log1p(price)`; report every final metric per price band.

**Baseline to beat.** A location-only baseline (region median, state fallback), evaluated out-of-fold with `GroupKFold(5)` on `group_id`: MAE $318, RMSE $518, median absolute error 18.8%, R² 0.468 on log rent (consistent with region η² 48.1% in-sample). Saved in `reports/baseline_location_only.csv`.

**Regression to the mean.** Grouped by true rent, the baseline's bias runs from +35.7% ($250–$750) to −58.2% ($4,000+), and it never predicts above $4,000 (the highest region median is about $2,900). Grouped by predicted rent, it is unbiased (−1.7% to +0.2%) with a typical error of 18–21%. The final model is evaluated both ways: the true-band slope should flatten, and the predicted-band bias should stay near zero.

**Rare groups.** Every category keeps its own one-hot column in every CV fold (valet parking ≈ 86 listings per learning fold). The riskiest rare groups (baseline bias ≥ 15%): homes over 3,000 sq ft (−52.4%), 4+ baths (−34.1%), 5+ beds (−33.3%), valet parking (−23.9%), in-law units (+24.9%), lofts (−15.7%).

**Geographic imbalance.** Listings are concentrated (Gini 0.53; the largest 10% of regions hold 32.8%). Baseline error rises from 18.5% in regions with 500+ listings to 25.1% in regions with fewer than 30 (4.3% of listings are in regions under 100). The worst regions are mixed resort/rural markets (Palm Springs, Rockies, Santa Fe), small rural regions (Clovis, Scottsbluff, Roswell) and a college town (Morgantown).

**Student-layout homes** (442; 4+ beds, baths ≥ beds): the worst group (typical error 59.8%). Their baseline bias (+29.2%) is opposite to other 4+ bedroom homes (−28.9%). They mix per-room student lets (46.2% under $750) and large luxury homes (28.7% above $2,500), concentrated in San Marcos, Gainesville, Tallahassee, Denver and Tippecanoe. An indicator feature helps only in combination with location and size.

**Error-analysis plan.** 23 groups (true and predicted price bands, region-size bands, large-home groups, rare categories) are fixed before the final evaluation in `reports/error_analysis_plan.csv`.