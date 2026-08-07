from __future__ import annotations

import pandas as pd
from common import config_argument

from favorita_forecasting.utils import read_json


def pct(value) -> str:
    return "n/a" if pd.isna(value) else f"{float(value):.2%}"


if __name__ == "__main__":
    _, cfg = config_argument("Generate an executive Markdown report from pipeline artifacts")
    champion = read_json(cfg.path("artifacts_dir") / "champion_spec.json")
    card = read_json(cfg.path("artifacts_dir") / "model_card.json")
    holdout = pd.read_csv(cfg.path("outputs_dir") / "holdout_metrics.csv").iloc[0]
    validation = pd.read_csv(cfg.path("outputs_dir") / "validation_summary.csv")
    hierarchy = pd.read_csv(cfg.path("outputs_dir") / "holdout_hierarchy_metrics.csv")
    inventory_path = cfg.path("outputs_dir") / "inventory_comparison.csv"
    inventory = pd.read_csv(inventory_path) if inventory_path.exists() else pd.DataFrame()

    best_baseline = validation[validation["model"].isin(["seasonal_naive_7", "four_week_mean", "dow_mean"])].sort_values("mean_wape").head(1)
    baseline_wape = float(best_baseline.iloc[0]["mean_wape"]) if len(best_baseline) else float("nan")
    validation_gain = (baseline_wape - float(champion["validation_mean_wape"])) / baseline_wape if baseline_wape > 0 else float("nan")

    inventory_text = "Simulation non exécutée."
    if not inventory.empty:
        inventory_text = inventory.to_markdown(index=False)

    report = f"""# Rapport exécutif — Favorita Retail Forecasting

## Décision

Le champion gelé est **{champion['model']}** avec les paramètres `{champion.get('params', {})}`. Il a été choisi uniquement sur les fenêtres de validation temporelle, avec la WAPE comme métrique principale et le biais comme garde-fou.

## Résultats

| Indicateur | Valeur |
|---|---:|
| WAPE validation champion | {pct(champion['validation_mean_wape'])} |
| Gain validation vs meilleure baseline | {pct(validation_gain)} |
| WAPE holdout final | {pct(holdout['wape'])} |
| Biais holdout final | {pct(holdout['bias'])} |
| MAE holdout | {float(holdout['mae']):.2f} |
| RMSLE holdout | {float(holdout['rmsle']):.3f} |

## Lecture métier

- Une WAPE plus faible indique une réduction du volume total d'erreur.
- Un biais négatif indique une sous-prévision globale et donc un risque de rupture.
- Les métriques par famille, magasin, promotion et horizon se trouvent dans `outputs/`.
- Les prévisions bottom-up sont cohérentes avec les agrégats géographiques et produit.

## Performance hiérarchique

{hierarchy.to_markdown(index=False)}

## Simulation supply chain

{inventory_text}

## Limites

- Les ventes observées sont un proxy de la demande.
- Les stocks, ruptures, délais et coûts réels ne sont pas fournis.
- `onpromotion` ne contient pas la profondeur de remise.
- La simulation d'inventaire repose sur des hypothèses explicites.
- Le périmètre portfolio ne représente pas nécessairement tous les magasins et familles.

## Traçabilité

- Holdout : {champion['holdout']['forecast_start']} au {champion['holdout']['forecast_end']}.
- Modèle et métriques : `artifacts/model_card.json`.
- Configuration : `config/project.yaml`.
- Tests : `pytest`.
"""
    output = cfg.path("reports_dir") / "EXECUTIVE_REPORT.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")
    print(output)
