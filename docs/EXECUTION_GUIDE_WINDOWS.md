# Guide d'exécution Windows / VS Code

## 1. Préparer l'environnement

Dans PowerShell, à la racine du projet :

```powershell
py -3.12 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e .
pip install -r requirements-core.txt
```

Installe ensuite les modèles et applications :

```powershell
pip install -r requirements-models.txt
pip install -r requirements-apps.txt
```

LightGBM et CatBoost sont CPU. Aucun GPU n'est requis.

## 2. Tester rapidement

```powershell
python scripts\run_demo.py
pytest
```

## 3. Ajouter les données Kaggle

Place dans `data\raw` les fichiers suivants, compressés ou extraits :

```text
train.csv(.7z)
test.csv(.7z)
items.csv(.7z)
stores.csv(.7z)
oil.csv(.7z)
holidays_events.csv(.7z)
transactions.csv(.7z)
sample_submission.csv(.7z)
```

Puis :

```powershell
python scripts\00_extract_archives.py --config config\project.yaml
```

## 4. Exécuter les étapes

```powershell
python scripts\01_audit_raw_data.py --config config\project.yaml
python scripts\02_select_scope.py --config config\project.yaml
python scripts\03_build_curated_panel.py --config config\project.yaml
python scripts\04_statistical_analysis.py --config config\project.yaml
python scripts\05_build_feature_dataset.py --config config\project.yaml
python scripts\05b_feature_diagnostics.py --config config\project.yaml
python scripts\05c_build_family_weights.py --config config\project.yaml
python scripts\06_backtest_and_select.py --config config\project.yaml
python scripts\06b_statistical_models_benchmark.py --config config\project.yaml
python scripts\07_evaluate_holdout.py --config config\project.yaml
python scripts\08_train_final_and_predict.py --config config\project.yaml
python scripts\09_inventory_simulation.py --config config\project.yaml
python scripts\10_explain_model.py --config config\project.yaml
python scripts\11_generate_executive_report.py --config config\project.yaml
```

## 5. Temps et mémoire

Le premier agrégat DuckDB lit le gros CSV sans le charger entièrement dans pandas. Le panel final est limité à quelques magasins et familles. Ne monte pas immédiatement à 54 magasins × toutes les familles.

Commence avec la configuration fournie. Après validation, augmente progressivement `n_stores` et `n_families`.

## 6. Résultats attendus

- `outputs/raw_file_audit.csv`
- `artifacts/scope_manifest.json`
- `data/processed/curated_panel.parquet`
- `data/processed/features.parquet`
- `outputs/statistical_analysis/`
- `outputs/validation_results.csv`
- `artifacts/champion_spec.json`
- `outputs/holdout_predictions.parquet`
- `outputs/holdout_metrics.csv`
- `outputs/test_forecasts.parquet`
- `outputs/inventory_comparison.csv`
- `artifacts/model_card.json`

## 7. Lancer les applications

```powershell
uvicorn favorita_forecasting.api.main:app --reload
streamlit run src\favorita_forecasting\dashboard\app.py
```
