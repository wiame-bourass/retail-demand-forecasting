# Rapport exécutif — Favorita Retail Forecasting

## Décision

Le champion gelé est **hist_gradient_boosting** avec les paramètres `{'l2_regularization': 0.0, 'learning_rate': 0.1, 'max_iter': 200, 'max_leaf_nodes': 31}`. Il a été choisi uniquement sur les fenêtres de validation temporelle, avec la WAPE comme métrique principale et le biais comme garde-fou.

## Résultats

| Indicateur | Valeur |
|---|---:|
| WAPE validation champion | 8.73% |
| Gain validation vs meilleure baseline | 22.12% |
| WAPE holdout final | 14.00% |
| Biais holdout final | 8.58% |
| MAE holdout | 319.54 |
| RMSLE holdout | 0.222 |

## Lecture métier

- Une WAPE plus faible indique une réduction du volume total d'erreur.
- Un biais négatif indique une sous-prévision globale et donc un risque de rupture.
- Les métriques par famille, magasin, promotion et horizon se trouvent dans `outputs/`.
- Les prévisions bottom-up sont cohérentes avec les agrégats géographiques et produit.

## Performance hiérarchique

| level        |       mae |      wape |      bias |    rmsle |      rmse |   forecast_accuracy |    n |   groups |
|:-------------|----------:|----------:|----------:|---------:|----------:|--------------------:|-----:|---------:|
| total        | 19063.8   | 0.0938366 | 0.0858031 | 0.104736 | 24319.8   |            0.906163 |   16 |        1 |
| state        |  7300.89  | 0.10781   | 0.0858031 | 0.128475 | 14178.4   |            0.89219  |   48 |        3 |
| city         |  5686.18  | 0.111955  | 0.0858031 | 0.125303 | 11539.3   |            0.888045 |   64 |        4 |
| store        |  2898.77  | 0.114147  | 0.0858031 | 0.130126 |  4068.23  |            0.885853 |  128 |        8 |
| family       |  1694.99  | 0.100118  | 0.0858031 | 0.146497 |  3423.82  |            0.899882 |  192 |       12 |
| state_family |   744.719 | 0.124633  | 0.0858031 | 0.264474 |  2157.5   |            0.875367 |  544 |       34 |
| city_family  |   586.43  | 0.129895  | 0.0858031 | 0.254855 |  1775.68  |            0.870105 |  720 |       45 |
| store_family |   319.539 | 0.139984  | 0.0858031 | 0.222244 |   689.649 |            0.860016 | 1424 |       89 |

## Simulation supply chain

| policy           |   demand_units |   fulfilled_units |   lost_sales_units |   holding_cost |   stockout_cost |   total_cost |   orders |   fill_rate |
|:-----------------|---------------:|------------------:|-------------------:|---------------:|----------------:|-------------:|---------:|------------:|
| champion         |    3.25055e+06 |       3.25026e+06 |            294.121 |         407944 |         588.241 |       408532 |     1049 |    0.99991  |
| seasonal_naive_7 |    3.25055e+06 |       3.24942e+06 |           1134.55  |         339494 |        2269.1   |       341763 |     1019 |    0.999651 |

## Limites

- Les ventes observées sont un proxy de la demande.
- Les stocks, ruptures, délais et coûts réels ne sont pas fournis.
- `onpromotion` ne contient pas la profondeur de remise.
- La simulation d'inventaire repose sur des hypothèses explicites.
- Le périmètre portfolio ne représente pas nécessairement tous les magasins et familles.

## Traçabilité

- Holdout : 2017-07-31 au 2017-08-15.
- Modèle et métriques : `artifacts/model_card.json`.
- Configuration : `config/project.yaml`.
- Tests : `pytest`.
