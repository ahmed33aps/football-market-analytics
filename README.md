# Football market analytics

An integrated football-analysis project with exploratory analysis, player-value regression, clustering, fuzzy logic and a Streamlit dashboard.

## Contents

- `Dashboard.py`
- `EDA&preprocssing_GA.ipynb`
- `FINAL.ipynb`
- `Data_for_Clustering.xlsx`
- `Full_data_for_analysis.csv`
- `premier_league.csv`
- `market_value_log_transform.png`

## Run

First restore the locally required data described in `PUBLIC_DATA_SETUP.md`.

Use Python 3.11+ in a virtual environment, then install dependencies:

```sh
python -m pip install -r requirements.txt
jupyter notebook
```

Open a notebook from this repository and run it from the repository root.

## Review notes

- Organized from the supplied project archives; duplicate folders, cache files and assignment handouts were not counted as separate projects.
- Saved notebook outputs were cleared so old results are not mistaken for a current successful run.
- Raw datasets are not included in this public source copy. See `PUBLIC_DATA_SETUP.md` for exact local filenames and expected columns. No new license is assigned.
- This is a portfolio/learning project, not a validated decision system. See `VALIDATION.md` for checks and remaining limits.
- The dashboard, regression and clustering are parts of one project. The dashboard rebuilds a deterministic local model from supplied CSV data. No uploaded pickle objects are loaded. Run `streamlit run Dashboard.py` from this directory.
