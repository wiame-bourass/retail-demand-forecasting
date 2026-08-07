# Retail Demand Forecasting

End-to-end retail sales forecasting project built on the **Corporación Favorita Grocery Sales Forecasting** dataset.

The project transforms item-level daily sales into a business-oriented `store × product family` forecasting problem and covers the full workflow from data processing to model evaluation, API exposure, dashboarding, Docker, and continuous integration.

## Problem

Forecast daily observed sales for each `store_nbr × family` combination over a **16-day horizon** using:

- historical sales;
- promotions;
- calendar and holiday information;
- store attributes;
- product-family information;
- oil-price signals;
- causal lag and rolling features.

Observed sales are treated as a proxy for demand because the dataset does not provide inventory levels, stockouts, or unmet demand.

## Data

The original Favorita training set contains more than **125 million item-store-day observations**.

To keep the project computationally practical while preserving meaningful retail structure, the raw data are aggregated as:

```text
date × store_nbr × item_nbr
          + item metadata
                ↓
date × store_nbr × item_nbr × family
                ↓
           aggregation
                ↓
date × store_nbr × family
```

DuckDB is used for memory-efficient processing of the large CSV files, followed by Parquet for downstream stages.

Raw Kaggle data are **not stored in this repository**.

## Pipeline

```text
Raw Favorita data
      ↓
Data audit & cleaning
      ↓
Scope selection
      ↓
Store-family aggregation
      ↓
Statistical analysis
      ↓
Causal feature engineering
      ↓
Temporal backtesting
      ↓
Model selection
      ↓
Final 16-day holdout
      ↓
Error analysis
      ↓
Inventory simulation
      ↓
FastAPI + Streamlit
      ↓
Docker + GitHub Actions
```

## Methodology

### Leakage prevention

All target-derived features use past information only.

Example:

```python
sales.shift(1).rolling(7).mean()
```

The current target value is never included in the features used to predict that same date.

### Temporal validation

Random train/test splits are not used.

- historical 16-day windows are used for model selection;
- the final 16 days are kept as an untouched holdout;
- the holdout is evaluated only after the champion model is frozen;
- multi-step forecasts are generated recursively to reproduce production conditions.

### Baselines

The forecasting models are compared against simple but strong references such as:

- seasonal naïve forecast using lag 7;
- four-week historical mean;
- historical day-of-week mean.

A more complex model is retained only if it improves on these references.

## Final results

The selected model is **HistGradientBoosting**.

| Metric | Result |
|---|---:|
| Validation WAPE | **8.73%** |
| Improvement vs best validation baseline | **22.12%** |
| Final holdout WAPE | **14.00%** |
| Holdout normalized bias | **+8.58%** |
| Holdout MAE | **319.54** |
| Holdout RMSLE | **0.222** |

The final holdout covers **2017-07-31 to 2017-08-15** and was never used for model selection or hyperparameter tuning.

The positive bias indicates that the model tends to **overforecast globally**, which is important when interpreting downstream inventory results.

More detailed metrics by store, family, promotion status, forecast horizon, and hierarchy level are available in the generated outputs and report.

## Metrics

**WAPE** is the primary model-selection metric because it expresses total absolute forecasting error relative to total sales volume.

**Bias** is used as a guardrail because a model can have acceptable WAPE while systematically over- or under-forecasting.

**MAE** expresses the average error directly in sales units.

**RMSLE** complements volume-based metrics by giving more importance to proportional errors across series of different sizes.

## Inventory simulation

Forecasts are also evaluated through a simplified inventory simulation using assumptions for:

- initial stock;
- lead time;
- review period;
- safety stock;
- holding cost;
- stockout cost.

This is a **scenario analysis**, not a reconstruction of Favorita's real supply-chain policy, because the original dataset does not contain inventory or stockout information.

## Run the application with Docker

For users who only want to explore the application, the repository contains the small precomputed runtime artifacts required by FastAPI and Streamlit.

Requirements:

- Git
- Docker

```bash
git clone https://github.com/wiame-bourass/retail-demand-forecasting.git
cd retail-demand-forecasting
docker compose up --build
```

Then open:

- FastAPI documentation: `http://localhost:8000/docs`
- Streamlit dashboard: `http://localhost:8501`

No local Python environment and no raw Favorita dataset are required for this Docker demo.

## Reproduce the full ML pipeline

To reproduce training and evaluation from the original data:

1. Download the original Favorita competition files.
2. Place the extracted CSV files in `data/raw/`.
3. Create a Python 3.12 environment and install the project dependencies.
4. Run the pipeline.

Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt
.\run_project.ps1
```

The pipeline generates processed datasets, model artifacts, forecasts, evaluation tables, monitoring outputs, and the executive report.

## API and dashboard

The application layer exposes precomputed batch forecasts rather than retraining the model on each request.

FastAPI provides endpoints for:

- service health;
- model metadata;
- forecast retrieval;
- forecast aggregation.

Streamlit provides an interactive view of model metrics, forecasts, error analysis, and inventory results.

## Testing and CI

The project includes automated tests for:

- forecasting metrics;
- temporal split logic;
- causal feature construction;
- leakage prevention;
- hierarchical consistency;
- intermittent-demand models;
- inventory simulation;
- monitoring utilities.

GitHub Actions runs on every push and pull request to:

```text
install dependencies
      ↓
run Ruff
      ↓
run pytest
      ↓
build the Docker image
```

## Repository structure

```text
.
├── .github/workflows/       # continuous integration
├── artifacts/               # lightweight runtime metadata
├── config/                  # pipeline configuration
├── docs/                    # methodology
├── notebooks/               # exploratory analysis
├── outputs/                 # lightweight runtime outputs
├── reports/                 # results and figures
├── scripts/                 # executable pipeline stages
├── src/favorita_forecasting/
│   ├── api/
│   ├── dashboard/
│   ├── data/
│   ├── evaluation/
│   ├── features/
│   ├── modeling/
│   └── simulation/
├── tests/
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
└── run_project.ps1
```

## Main limitations

- observed sales are not identical to unconstrained demand;
- stock levels and stockouts are unavailable;
- promotion depth and marketing intensity are unavailable;
- inventory costs and lead times are simulated assumptions;
- the portfolio scope does not cover every possible store-family combination;
- the model operates at `store × family` level rather than the original Kaggle `store × item` level.

## Tech stack

- Python
- DuckDB
- Pandas
- scikit-learn
- Parquet
- FastAPI
- Streamlit
- Docker
- Pytest
- Ruff
- GitHub Actions (CI/CD)

## Documentation

Detailed methodological choices are documented in:

- `docs/METHODOLOGY.md`
- `reports/EXECUTIVE_REPORT.md`
