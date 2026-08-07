# Commence ici

## Vérification rapide

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e .
pip install -r requirements-core.txt
python scripts\run_demo.py
```

## Projet réel

1. Mets les archives Kaggle dans `data\raw`.
2. Installe toutes les dépendances :

```powershell
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

3. Exécute `run_project.ps1`.
4. Ouvre `reports\EXECUTIVE_REPORT.md` et les notebooks.
5. Lance ensuite l'API et le dashboard.
