# U.S. Rental Price Estimation System

IT3051 Fundamentals of Data Mining – group mini project (Algoverse)

A machine-learning system that estimates the monthly rent of a U.S. residential property from its details (location, size, bedrooms, bathrooms, amenities). Built on the USA Housing Listings dataset (384,977 Craigslist listings).

---

## Project structure

```
fdm-rental-price/
├── data/                  # NOT in Git (too large) – see "Getting the data"
│   ├── raw/               # housing.csv (original download)
│   ├── interim/           # housing_raw.parquet (fast copy made by notebook 01)
│   └── processed/         # cleaned train/test files made by the preprocessing notebook
├── notebooks/
│   ├── 01_eda.ipynb                        # Steps 0–1: setup, structure of the data
│   ├── 02_eda_target.ipynb                 # Step 2: the target (price)
│   ├── 03_eda_missing_duplicates.ipynb     # Steps 3–4: missing values, duplicates
│   ├── 04_eda_validity.ipynb               # Step 5: validity checks
│   └── ...                                 # later EDA, preprocessing, modelling
├── src/                   # shared Python code, imported by notebooks and the backend
│   ├── __init__.py
│   └── utils.py           # paths, settings, confirmed cleaning limits, logging helpers
├── reports/
│   ├── figures/           # charts for the report
│   ├── observations_log.csv
│   └── data_dictionary.csv
├── models/                # NOT in Git – trained model files (.joblib)
├── backend/               # FastAPI prediction API
├── frontend/              # web form for users
├── requirements.txt
└── README.md
```

---

## Setup (do this once)

### 1. Clone the repository

```bash
git clone https://github.com/<owner>/fdm-rental-price.git
cd fdm-rental-price
```

### 2. Create a virtual environment and install packages

A virtual environment keeps this project's packages separate from everything else on your computer, so everyone in the group runs the same versions.

**Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

**Mac / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Register the environment as a Jupyter kernel

```bash
python -m ipykernel install --user --name fdm-rental --display-name "FDM rental (.venv)"
```

In Jupyter or VS Code, choose the **FDM rental (.venv)** kernel for every notebook.

### 4. Getting the data

The dataset is too large for GitHub. Download `housing.csv` from the group's shared Google Drive folder and place it at:

```
data/raw/housing.csv
```

Then run `notebooks/01_eda.ipynb` once. It creates the fast parquet copy that all later notebooks load.

---

## How we work together

### Golden rules

1. **Never commit directly to `main`.** Work on your own branch and open a pull request.
2. **One person edits a notebook at a time.** Notebooks are JSON files, and two people editing the same notebook almost always produces a merge conflict that is painful to fix. Each notebook has an owner (see the task table below).
3. **Shared code goes in `src/`**, not copied between notebooks.
4. **Never commit data or model files.** `.gitignore` blocks them, so don't force-add them.
5. **Pull before you start working** each day, so you build on everyone's latest work.

### Daily workflow

```bash
# 1. Start from the latest main
git checkout main
git pull

# 2. Create a branch for your task (name: yourname/short-description)
git checkout -b mihijith/step6-outliers

# 3. Work, then save your progress in small commits
git add notebooks/05_eda_outliers.ipynb src/utils.py
git commit -m "Step 6: outlier analysis for sqfeet and price per sqft"

# 4. Push your branch to GitHub
git push -u origin mihijith/step6-outliers
```

Then on GitHub, open a **Pull Request** from your branch into `main`. At least one teammate reviews it and approves it, then it is merged.

### Keeping your branch up to date

If `main` changed while you were working:

```bash
git checkout main
git pull
git checkout mihijith/step6-outliers
git merge main
```

### About `reports/observations_log.csv`

Each notebook clears and rewrites only its own step's entries, but the file itself is shared, so two branches may both change it. If you get a merge conflict on it, accept either version and **re-run the notebooks for the affected steps**. The log is always regenerated from the notebooks.

### Good commit messages

Say *what* changed and *why*, in the present tense:

- ✅ `Add sale-listing detection to Step 5 validity checks`
- ✅ `Move confirmed sqfeet limits into utils.py`
- ❌ `update`
- ❌ `fixed stuff`

---

## Task ownership

| Area | Owner | Files |
|---|---|---|
| EDA Steps 0–5 | | notebooks 01–04 |
| EDA Steps 6+ | | |
| Cleaning and preprocessing code | | `src/cleaning.py`, `src/features.py` |
| Model: Linear Regression (Ridge) | | |
| Model: k-Nearest Neighbours | | |
| Model: Random Forest | | |
| Model: LightGBM | | |
| Model: MLP | | |
| Backend API | | `backend/` |
| Frontend | | `frontend/` |
| Report | | |

---

## Running the prediction system

*(To be completed once the backend and frontend are built.)*
