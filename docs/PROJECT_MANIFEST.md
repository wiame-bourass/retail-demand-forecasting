# Manifeste du dépôt

## Configuration

- `config/project.yaml` : exécution sur les données Kaggle.
- `config/demo.yaml` : smoke test rapide.

## Données

- `scripts/00_extract_archives.py` : extraction des fichiers `.7z`.
- `scripts/01_audit_raw_data.py` : schémas, volumes, dates, doublons.
- `scripts/02_select_scope.py` : magasins et familles représentatifs.
- `scripts/03_build_curated_panel.py` : jointure article-famille, agrégation et grille.

## Analyse

- `scripts/04_statistical_analysis.py` : intermittence, saisonnalité, promotions, jours fériés, pétrole.
- `scripts/06b_statistical_models_benchmark.py` : ETS, SARIMA, Croston et TSB sur séries représentatives.
- `sql/eda_queries.sql` et `sql/quality_checks.sql` : analyses SQL DuckDB.

## Features

- `scripts/05_build_feature_dataset.py` : features causales.
- `scripts/05b_feature_diagnostics.py` : manquants, constantes, redondances, information mutuelle.
- `tests/test_features_no_leakage.py` : preuve anti-fuite.

## Modèles

- `scripts/06_backtest_and_select.py` : réglage sur validation seulement.
- `scripts/07_evaluate_holdout.py` : évaluation finale gelée.
- `scripts/08_train_final_and_predict.py` : entraînement complet et test Kaggle agrégé.
- `scripts/10_explain_model.py` : importance par permutation sur données d'entraînement.

## Business et supply chain

- `scripts/09_inventory_simulation.py` : coût, fill rate et ventes perdues simulées.
- `scripts/11_generate_executive_report.py` : rapport automatique.
- `docs/INTERVIEW_STORY.md` : présentation orale.

## Industrialisation

- `src/favorita_forecasting/api/main.py` : FastAPI.
- `src/favorita_forecasting/dashboard/app.py` : Streamlit.
- `Dockerfile`, `docker-compose.yml`, `.github/workflows/ci.yml`.
- `deploy/` et `docs/GCP_DEPLOYMENT.md`.
