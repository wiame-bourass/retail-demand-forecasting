# Rapport exécutif — Favorita Retail Forecasting

## Décision

Le champion gelé est **hist_gradient_boosting** avec les paramètres `{'l2_regularization': 0.0, 'learning_rate': 0.08, 'max_iter': 100, 'max_leaf_nodes': 31}`. Il a été choisi uniquement sur les fenêtres de validation temporelle, avec la WAPE comme métrique principale et le biais comme garde-fou.

## Résultats

| Indicateur | Valeur |
|---|---:|
| WAPE validation champion | 17.38% |
| Gain validation vs meilleure baseline | 14.11% |
| WAPE holdout final | 16.31% |
| Biais holdout final | -4.68% |
| MAE holdout | 4.61 |
| RMSLE holdout | 0.238 |

## Lecture métier

- Une WAPE plus faible indique une réduction du volume total d'erreur.
- Un biais négatif indique une sous-prévision globale et donc un risque de rupture.
- Les métriques par famille, magasin, promotion et horizon se trouvent dans `outputs/`.
- Les prévisions bottom-up sont cohérentes avec les agrégats géographiques et produit.

## Performance hiérarchique

| level        |      mae |      wape |       bias |     rmsle |     rmse |   forecast_accuracy |   n |   groups |
|:-------------|---------:|----------:|-----------:|----------:|---------:|--------------------:|----:|---------:|
| total        | 25.4506  | 0.0749788 | -0.0468204 | 0.0816423 | 28.2938  |            0.925021 |  16 |        1 |
| state        |  9.8913  | 0.0874208 | -0.0468204 | 0.107161  | 12.3224  |            0.912579 |  48 |        3 |
| city         |  9.8913  | 0.0874208 | -0.0468204 | 0.107161  | 12.3224  |            0.912579 |  48 |        3 |
| store        |  9.8913  | 0.0874208 | -0.0468204 | 0.107161  | 12.3224  |            0.912579 |  48 |        3 |
| family       |  9.14679 | 0.107788  | -0.0468204 | 0.141961  | 11.9092  |            0.892212 |  64 |        4 |
| state_family |  4.61415 | 0.163122  | -0.0468204 | 0.238077  |  5.98966 |            0.836878 | 192 |       12 |
| city_family  |  4.61415 | 0.163122  | -0.0468204 | 0.238077  |  5.98966 |            0.836878 | 192 |       12 |
| store_family |  4.61415 | 0.163122  | -0.0468204 | 0.238077  |  5.98966 |            0.836878 | 192 |       12 |

## Simulation supply chain

| policy           |   demand_units |   fulfilled_units |   lost_sales_units |   holding_cost |   stockout_cost |   total_cost |   orders |   fill_rate |
|:-----------------|---------------:|------------------:|-------------------:|---------------:|----------------:|-------------:|---------:|------------:|
| champion         |           5431 |              5431 |                  0 |        563.848 |               0 |      563.848 |      173 |           1 |
| seasonal_naive_7 |           5431 |              5431 |                  0 |        587.894 |               0 |      587.894 |      163 |           1 |

## Limites

- Les ventes observées sont un proxy de la demande.
- Les stocks, ruptures, délais et coûts réels ne sont pas fournis.
- `onpromotion` ne contient pas la profondeur de remise.
- La simulation d'inventaire repose sur des hypothèses explicites.
- Le périmètre portfolio ne représente pas nécessairement tous les magasins et familles.

## Traçabilité

- Holdout : 2021-03-16 au 2021-03-31.
- Modèle et métriques : `artifacts/model_card.json`.
- Configuration : `config/project.yaml`.
- Tests : `pytest`.
