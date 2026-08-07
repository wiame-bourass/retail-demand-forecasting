# Plan de monitoring

## Qualité des données

- présence des fichiers ;
- schémas et clés ;
- fraîcheur de date ;
- doublons ;
- taux de valeurs manquantes ;
- promotions supérieures au catalogue ;
- ventes historiques manquantes ;
- cible future accidentellement renseignée.

## Dérive

Le script `12_monitoring_snapshot.py` compare les 90 jours de référence au holdout :

- PSI des variables numériques ;
- nouvelles catégories ;
- variation du taux de valeurs manquantes.

Seuils initiaux : PSI > 0,25, nouvelles catégories > 5 %, écart de missingness > 10 points.

## Performance

Après disponibilité des ventes réelles :

- WAPE, MAE, RMSLE et biais ;
- promotion/hors promotion ;
- horizon 1 à 16 ;
- magasin et famille ;
- coût simulé et fill rate.

## Déclencheurs de réentraînement

- dérive persistante ;
- dégradation WAPE significative ;
- biais dépassant le garde-fou ;
- changement d'assortiment ou de taxonomie ;
- nouvelles pratiques promotionnelles.
