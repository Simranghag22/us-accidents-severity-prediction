# US-Traffic-Accident-Hotspot-Risk-Factor-Analysis

Analyzes 7.7M US traffic accidents (2016–2023) to uncover the temporal, weather, and infrastructure factors behind accident frequency and severity. Extends the EDA into a leakage-audited XGBoost model predicting severe-accident risk, explained via SHAP, plus K-Means hotspot risk-tier clustering.

**Folder:** [`US_Traffic_Accident_Hotspot_Risk_Factor_Analysis/`](./US_Traffic_Accident_Hotspot_Risk_Factor_Analysis)

---

## 🔗 Quick Links

| Resource | Link |
|---|---|
| 📓 Notebook | [`01_Notebook/US_Traffic_Accident_Hotspot_Risk_Factor_Analysis_ML_Extension.ipynb`](./US_Traffic_Accident_Hotspot_Risk_Factor_Analysis/01_Notebook/US_Traffic_Accident_Hotspot_Risk_Factor_Analysis_ML_Extension.ipynb) |
| 📄 Full Project Report | [`02_Project_Report/Report-US_Traffic_Accident_Hotspot_Risk_Factor_Analysis_Report.pdf`](./US_Traffic_Accident_Hotspot_Risk_Factor_Analysis/02_Project_Report/Report-US_Traffic_Accident_Hotspot_Risk_Factor_Analysis_Report.pdf) |
| 📁 Dataset | [US Accidents (2016–2023) — Kaggle](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents) *(not included in this repo — see note below)* |
| 🎓 Internship Certificate (Corizo) | [Verify Credential](https://credentials.corizo.in/credential/02e62977-01dc-4caa-8c76-be15f44113ce) |
| 🎓 Training Certificate (Corizo) | [Verify Credential](https://credentials.corizo.in/credential/b5774572-2577-4947-abd0-bba4dc23a0e4) |

> **Note on size:** the raw dataset (~7.7M rows, ~6 GB) and the pipeline's intermediate checkpoint files (cleaned/optimized/feature-engineered Parquet snapshots and the trained XGBoost `joblib` artifact) are not included in this repo — they're regenerable directly from the notebook and would otherwise push the repo well past a reasonable upload size. Link out to the Kaggle source instead of downloading a local copy.

---

## Overview

Road safety agencies, insurers, and navigation apps all need the same underlying answer: *where, when, and under what conditions are accidents most likely to be severe?* This project builds that answer end-to-end on the US Accidents dataset, a countrywide, timestamped accident log spanning seven years:

- **Exploratory Analysis (Part 1):** examines how accident frequency and severity vary by time of day, day of week, season, weather, visibility, wind, road infrastructure, and geography — closing with a nationwide hotspot density map.
- **Severity Prediction (Part 2):** reframes the EDA's patterns as a supervised learning problem — a leakage-audited XGBoost classifier that estimates the probability a given accident, if it occurs, will be severe (Severity 3–4), using only conditions knowable at or before the moment of the accident.
- **Risk-Tier Clustering (Bonus):** an unsupervised K-Means layer on top of the hotspot map that separates *high-volume* hotspots from *high-severity-rate* hotspots — two meaningfully different kinds of "risky" location.

## Dataset

- **Source:** US Accidents (2016–2023) — Moosavi, S., Samavatian, M. H., Parthasarathy, S., & Ramnath, R. (Kaggle)
- **Coverage:** 49 US states, February 2016 – March 2023
- **Size (raw):** 7,728,394 records × 46 features
- **Size (cleaned & feature-engineered):** 6,985,002 records × 49 columns
- **Key feature groups:** accident info (severity, start/end time), geographic (lat/lng, city/county/state), weather (temperature, humidity, visibility, wind, precipitation), road infrastructure (junctions, traffic signals, crossings, railways, stops), and 12 engineered time/severity features (`Accident_Duration_Min`, `Day_of_Week`, `Weekend`, `Season`, `Time_of_Day`, `Rush_Hour`, `Severe_Accident`, etc.)

## Repository Structure

```
US_Traffic_Accident_Hotspot_Risk_Factor_Analysis/
├── 01_Notebook/
│   └── US_Traffic_Accident_Hotspot_Risk_Factor_Analysis_ML_Extension.ipynb
├── 02_Project_Report/
│   └── Report-US_Traffic_Accident_Hotspot_Risk_Factor_Analysis_Report.pdf
└── 03_Certificates/
    ├── Simran_Subhash_Ghag_Internship_Certificate_Corizo.pdf
    ├── Simran_Subhash_Ghag_Training_Certificate_Corizo.pdf
    ├── Internship_Certificate.png
    └── Training_Certificate.png
```

*(The raw dataset and pipeline checkpoint files are intentionally excluded — see the size note above.)*

## Workflow

Raw CSV load → data-quality checks (0 missing after cleaning, 0 duplicates) → column pruning (irrelevant/redundant/overly granular fields dropped) → missing-value handling (median for numeric weather fields, mode for categorical, forward-fill for timestamps, row-drop for irreplaceable fields) → dtype optimization (categorical conversion, boolean encoding — **6 GB → 1.3 GB, ≈29% memory reduction**) → 12 engineered features (duration, day/weekend, season, time-of-day, rush-hour, visibility/wind level, weather category, severe-accident flag) → exploratory data analysis across temporal, environmental, infrastructural, and geospatial dimensions → **Part 2:** leakage audit → stratified 1M-row sample → preprocessing pipeline (scaling + one-hot encoding) → time-based train/test split (train 2016–2021, test 2022–2023) → cross-validated model comparison → final tuned XGBoost model → evaluation (precision/recall/F1, ROC-AUC, PR-AUC) → feature importance & SHAP explainability → K-Means risk-tier clustering on hotspot locations.

To keep the ~7.7M-row pipeline resumable across sessions, progress was checkpointed at five stages (post-cleaning, post-optimization, post-feature-engineering, post-train/test-split, and post-model-training) using Parquet and `joblib` snapshots.

## Severity Prediction — Model Comparison

Three candidate models were compared via 3-fold stratified cross-validation (150,000-row training subsample), scored on ROC-AUC:

| Model | Mean ROC-AUC | Std. Dev. |
|---|---|---|
| Logistic Regression | 0.747 | ± 0.001 |
| Random Forest | 0.770 | ± 0.002 |
| **XGBoost** | **0.781** | ± 0.001 |

XGBoost was selected as the final model and retrained on the full 2016–2021 training set (793,814 rows, 24.71% severe) with `scale_pos_weight = 3.05` to handle class imbalance, then evaluated on a genuinely out-of-time 2022–2023 holdout (206,186 rows, 7.90% severe):

| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| Non-Severe | 0.93 | 0.91 | 0.92 | 189,888 |
| Severe | 0.20 | 0.25 | 0.22 | 16,298 |

**Overall:** Accuracy 0.86 · ROC-AUC 0.702 · PR-AUC 0.165 (vs. a 0.079 random baseline)

Accuracy was deliberately not used as the headline metric — with roughly 1 in 13 holdout accidents being severe, a model that never predicts "severe" would still score ~92% accuracy while being operationally useless. The model was explained both globally (gain-based feature importance — `State`, `Traffic_Signal`, and `Year` dominate) and locally (SHAP values, showing how each feature pushed individual predictions toward or away from "severe").

## Risk-Tier Clustering (Bonus)

Accident locations were binned geographically (0.1° lat/lng) and clustered with K-Means on two dimensions — accident volume and severe-accident rate — to distinguish hotspots that are risky *because of volume* from those risky *because of severity*:

| Cluster Profile | Locations | Avg. Accidents / Location | Avg. Severe Rate |
|---|---|---|---|
| Low-volume, high-severity-rate | 6,070 | 196.6 | 44.9% |
| High-volume, elevated-severity-rate | 463 | 5,594.4 | 22.1% |
| Extreme-volume, moderate-severity-rate | 63 | 19,550.0 | 21.0% |
| Low-volume, low-severity-rate | 11,036 | 173.6 | 7.4% |

## Key Findings

- Traffic accidents are heavily concentrated in major metropolitan regions — the strongest clusters appear in Southern California, the Texas urban corridor, Florida, and the Northeast US, with California (1.57M), Florida (766K), and Texas (545K) the top three states by raw accident count.
- Accident frequency rises sharply during rush-hour commuting windows (~7–8 AM and ~4–5 PM), with Friday showing the highest overall accident count and weekends dropping off noticeably.
- Urban infrastructure — intersections, traffic signals, junctions — is a consistent risk factor, reflecting complex vehicle interactions and higher decision-making demand at these locations; very few accidents occur near speed bumps or roundabouts.
- Weather-related factors (rain, snow, low visibility) show clear seasonal patterns but are secondary to traffic-volume effects — most accidents still happen in clear conditions and high visibility, simply because that's when most driving happens.
- The trained XGBoost model's top predictors (state-level location, proximity to a traffic signal, and year) independently confirm the EDA's own geographic and infrastructural findings — strong evidence the model learned genuine signal rather than noise.
- Not all accident hotspots are the same *kind* of hotspot: risk-tier clustering separates a small number of extreme-volume locations (63 locations, ~19,550 accidents each) from a much larger set of lower-volume locations with a disproportionately high severe-accident rate (6,070 locations, 44.9% severe) — each warranting a different intervention.

## Tools & Technologies

Python · Pandas · NumPy · Matplotlib · Seaborn · Plotly · Folium · Scikit-learn · XGBoost · SHAP · joblib

## Author

**Simran Subhash Ghag**
Data Science Internship & Training — Corizo, in association with IIT Bombay's Mood Indigo

## License

`[add a license if you intend this repo to be public, e.g. MIT]`

---

## 🎓 Certificates

**Training Certificate** — Data Science (06 May 2026 – 17 June 2026)
Corizo Dice ID: CRZ158595 | [Verify Credential](https://credentials.corizo.in/credential/b5774572-2577-4947-abd0-bba4dc23a0e4)

![Training Certificate](./US_Traffic_Accident_Hotspot_Risk_Factor_Analysis/03_Certificates/Training_Certificate.png)

**Internship Certificate** — Data Science (17 June 2026 – 06 July 2026)
Corizo Dice ID: CRZ156098 | [Verify Credential](https://credentials.corizo.in/credential/02e62977-01dc-4caa-8c76-be15f44113ce)

![Internship Certificate](./US_Traffic_Accident_Hotspot_Risk_Factor_Analysis/03_Certificates/Internship_Certificate.png)
