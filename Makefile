PYTHON ?= python
CONFIG ?= config/project.yaml

.PHONY: install demo extract audit scope curate stats features weights backtest statistical holdout final inventory explain report monitor test api dashboard all

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e .
	$(PYTHON) -m pip install -r requirements.txt

demo:
	$(PYTHON) scripts/run_demo.py

extract:
	$(PYTHON) scripts/00_extract_archives.py --config $(CONFIG)

audit:
	$(PYTHON) scripts/01_audit_raw_data.py --config $(CONFIG)

scope:
	$(PYTHON) scripts/02_select_scope.py --config $(CONFIG)

curate:
	$(PYTHON) scripts/03_build_curated_panel.py --config $(CONFIG)

stats:
	$(PYTHON) scripts/04_statistical_analysis.py --config $(CONFIG)

features:
	$(PYTHON) scripts/05_build_feature_dataset.py --config $(CONFIG)
	$(PYTHON) scripts/05b_feature_diagnostics.py --config $(CONFIG)

weights:
	$(PYTHON) scripts/05c_build_family_weights.py --config $(CONFIG)

backtest:
	$(PYTHON) scripts/06_backtest_and_select.py --config $(CONFIG)

statistical:
	$(PYTHON) scripts/06b_statistical_models_benchmark.py --config $(CONFIG)

holdout:
	$(PYTHON) scripts/07_evaluate_holdout.py --config $(CONFIG)

final:
	$(PYTHON) scripts/08_train_final_and_predict.py --config $(CONFIG)

inventory:
	$(PYTHON) scripts/09_inventory_simulation.py --config $(CONFIG)

explain:
	$(PYTHON) scripts/10_explain_model.py --config $(CONFIG)

report:
	$(PYTHON) scripts/11_generate_executive_report.py --config $(CONFIG)

monitor:
	$(PYTHON) scripts/12_monitoring_snapshot.py --config $(CONFIG)

all: audit scope curate stats features weights backtest statistical holdout final inventory explain report monitor

test:
	pytest

api:
	uvicorn favorita_forecasting.api.main:app --reload --host 0.0.0.0 --port 8000

dashboard:
	streamlit run src/favorita_forecasting/dashboard/app.py
