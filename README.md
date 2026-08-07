# Retail Demand Forecasting

**Sujet :** prévision hiérarchique et groupée des ventes retail intégrant les promotions et les événements.

Ce dépôt transforme le dataset Kaggle original **Corporación Favorita Grocery Sales Forecasting** du niveau `date × store_nbr × item_nbr` vers le niveau métier `date × store_nbr × family`, puis construit un pipeline complet : audit, nettoyage, analyse statistique, feature engineering causal, validation temporelle, modèles globaux, analyse d'erreurs, agrégation hiérarchique, simulation de stock, API, dashboard, tests et suivi MLflow.


## Ce que ce projet démontre

* Traitement de données à grande échelle avec DuckDB
* Feature engineering pour les séries temporelles
* Validation temporelle et prévention des fuites de données
* Modèles de Gradient Boosting
* Évaluation hiérarchique
* Tests de Machine Learning
* Développement d’API avec FastAPI et d’interface avec Streamlit
* Conteneurisation avec Docker
* Intégration et déploiement continus (CI/CD)


## Granularité et limite métier

La cible est la somme des ventes observées au niveau `store_nbr × family`. Les ventes sont utilisées comme **proxy de la demande** : le dataset ne fournit ni stocks, ni ruptures, ni demande non satisfaite. La simulation supply chain utilise donc des hypothèses explicites.

## Démarrage rapide avec les données de démonstration

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e .
pip install -r requirements-core.txt
python scripts/run_demo.py
```

Le mode démo génère un mini-dataset Favorita compatible, exécute le pipeline avec `HistGradientBoosting`, produit les rapports et vérifie le fonctionnement avant le traitement des données Kaggle réelles.

## Exécution avec le dataset Kaggle réel

1. Télécharge les fichiers de la compétition originale.
2. Place les fichiers `.7z` ou `.csv` dans `data/raw/`.
3. Exécute :

```bash
python scripts/00_extract_archives.py --config config/project.yaml
python scripts/01_audit_raw_data.py --config config/project.yaml
python scripts/02_select_scope.py --config config/project.yaml
python scripts/03_build_curated_panel.py --config config/project.yaml
python scripts/04_statistical_analysis.py --config config/project.yaml
python scripts/05_build_feature_dataset.py --config config/project.yaml
python scripts/05b_feature_diagnostics.py --config config/project.yaml
python scripts/05c_build_family_weights.py --config config/project.yaml
python scripts/06_backtest_and_select.py --config config/project.yaml
python scripts/06b_statistical_models_benchmark.py --config config/project.yaml
python scripts/07_evaluate_holdout.py --config config/project.yaml
python scripts/08_train_final_and_predict.py --config config/project.yaml
python scripts/09_inventory_simulation.py --config config/project.yaml
python scripts/10_explain_model.py --config config/project.yaml
python scripts/11_generate_executive_report.py --config config/project.yaml
python scripts/12_monitoring_snapshot.py --config config/project.yaml
```

Ou :

```bash
make all CONFIG=config/project.yaml
```

## Ordre méthodologique

- Les fenêtres de validation servent au choix du modèle et des hyperparamètres.
- Les 16 derniers jours historiques constituent le holdout final et ne servent jamais au réglage.
- Le modèle champion est ensuite réentraîné sur tout l'historique et prédit les 16 jours Kaggle.
- Les poids de famille éventuels sont calculés uniquement avec la période d'entraînement.

## Fichiers clés

- `docs/METHODOLOGY_AND_MODEL_CHOICES.md` : démarche complète et justification des choix.
- `docs/EXECUTION_GUIDE_WINDOWS.md` : commandes exactes pour Windows/VS Code.
- `notebooks/` : analyses guidées, sans dupliquer la logique de production.
- `src/` : code réutilisable et testable.
- `sql/` : analyses SQL DuckDB.

## API et dashboard

Après l'étape finale :

```bash
uvicorn favorita_forecasting.api.main:app --reload
streamlit run src/favorita_forecasting/dashboard/app.py
```

## Tests

```bash
pytest
```

Les tests couvrent les métriques, la causalité des features, la cohérence bottom-up et la simulation de stock.

## Important : score Kaggle officiel

Le modèle principal prédit au niveau `store_nbr × family`. Il ne produit donc pas directement une soumission officielle Kaggle, qui exige des prévisions `store_nbr × item_nbr`. Le dépôt privilégie le cas d'usage métier agrégé. Une allocation famille → articles serait une extension séparée et devrait être évaluée avec la NWRMSLE officielle.
