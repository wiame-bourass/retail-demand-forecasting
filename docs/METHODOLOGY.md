# Méthodologie et justification des choix

## 1. Question métier

Prévoir les ventes quotidiennes observées de chaque famille de produits dans chaque magasin pour les 16 prochains jours, en exploitant l'historique, les promotions connues, les caractéristiques magasin, les événements, le prix du pétrole et la saisonnalité.

La vente observée est un proxy imparfait de la demande. Sans stock ni rupture, il est impossible d'estimer la demande non satisfaite. Cette limite est documentée et séparée de la simulation supply chain.

## 2. Pourquoi agréger au niveau famille

Le fichier brut est au niveau article. Un projet portfolio au niveau article ferait exploser le nombre de séries et le coût d'itération. Le niveau `store_nbr × family` conserve la diversité magasin-produit tout en permettant :

- un modèle global robuste ;
- une analyse promotionnelle lisible ;
- une structure hiérarchique géographique ;
- une simulation de stock interprétable ;
- une exécution réaliste sur une machine locale de 16 Go sans GPU.

La transformation est :

```text
date × store_nbr × item_nbr
        + items.csv
              ↓
date × store_nbr × item_nbr × family
              ↓ agrégation
date × store_nbr × family
```

## 3. Promotions au niveau famille

`onpromotion` est défini par article, magasin et date. Il n'est donc pas conservé comme simple booléen après agrégation. Les variables utilisées sont notamment :

- nombre d'articles en promotion ;
- indicateur d'au moins une promotion ;
- part des articles catalogue de la famille en promotion ;
- séquence de jours promotionnels ;
- jours depuis la dernière promotion ;
- moyenne historique causale des ventes sous/hors promotion ;
- uplift historique descriptif.

Cette information ne représente ni le montant de remise, ni l'intensité média, ni le budget de campagne.

## 4. Nettoyage

- Dates converties explicitement.
- Doublons contrôlés sur les clés attendues.
- `onpromotion` manquant traité comme faux selon la convention configurée.
- Ventes négatives isolées comme retours puis cible tronquée à zéro pour la prévision des ventes positives.
- Grille quotidienne complétée uniquement après la première observation d'une série active.
- Les absences historiques après activation sont interprétées comme zéro vente, avec indicateur de ligne imputée.

## 5. Analyse statistique utile

Les analyses ne se limitent pas à des graphiques descriptifs :

- intermittence par ADI et CV² ;
- autocorrélation hebdomadaire au lag 7 ;
- tendance linéaire robuste par série ;
- uplift promotionnel descriptif, test de Welch et taille d'effet ;
- effet jour de semaine ;
- effet jours fériés ;
- corrélation de Spearman entre pétrole et ventes agrégées ;
- segmentation volume / promotion / intermittence.

Les écarts promotionnels restent associatifs : les promotions ne sont pas assignées aléatoirement.

## 6. Features et absence de fuite

Toutes les statistiques construites à partir de la cible utilisent uniquement le passé :

```python
sales.shift(1).rolling(7).mean()
```

Les variables connues dans le futur peuvent être utilisées au jour prédit : calendrier, promotions du test, événements et pétrole fourni. Les transactions du jour futur ne sont jamais utilisées ; seules des versions retardées peuvent être activées.

Les principales features sont :

- lags 1, 7, 14, 21, 28, 364 ;
- moyennes, médianes et écarts-types mobiles décalés ;
- tendance récente 7 jours moins 28 jours ;
- calendrier et cyclical encodings ;
- promotions agrégées ;
- caractéristiques magasin et famille ;
- événements nationaux, régionaux et locaux ;
- pétrole et variations retardées ;
- transactions retardées, désactivées par défaut.

## 7. Validation temporelle

Aucun split aléatoire n'est autorisé.

1. Deux fenêtres historiques de validation de 16 jours servent à comparer les paramètres.
2. Les 16 derniers jours du train constituent le holdout final.
3. Le holdout n'est évalué qu'après gel du champion.
4. Le modèle final est réentraîné sur tout l'historique et prédit le test Kaggle.

La prévision est récursive : lorsque l'horizon dépasse un lag disponible, les prédictions antérieures sont réinjectées, comme en production.

## 8. Pourquoi ces baselines

- `lag 7` mesure la force de la saisonnalité hebdomadaire.
- moyenne des quatre mêmes jours des semaines précédentes réduit le bruit.
- moyenne historique par jour de semaine fournit une référence stable par série.

Un modèle avancé n'est retenu que s'il bat ces références sur des fenêtres futures.

## 9. Pourquoi LightGBM

LightGBM est le candidat principal car il :

- apprend des non-linéarités et interactions promotion-calendrier ;
- supporte bien les données tabulaires ;
- entraîne rapidement sur un panel global ;
- gère les valeurs manquantes ;
- fournit des importances et explications ;
- est plus scalable que des modèles statistiques séparés par série.

CatBoost sert de comparaison pour les catégories et HistGradientBoosting de contrôle scikit-learn. ETS/SARIMA ne sont appliqués qu'à un sous-ensemble car un modèle par série est coûteux et moins flexible avec les promotions.

## 10. Choix de la métrique

### WAPE — métrique principale

La WAPE mesure l'erreur absolue totale relativement au volume total. Elle est lisible par le métier et pénalise les erreurs sur les volumes importants.

### Biais — garde-fou supply chain

Le biais normalisé distingue sous-prévision et sur-prévision. Une faible WAPE avec forte sous-prévision peut provoquer des ruptures.

### MAE

La MAE exprime l'erreur en unités de vente et facilite l'interprétation opérationnelle.

### RMSLE

La RMSLE réduit la domination des très gros volumes et pénalise les erreurs relatives sur les petites séries.

### Weighted RMSLE famille

Une version expérimentale peut utiliser un poids de famille dérivé des poids article et des volumes historiques du train. Elle n'est pas appelée NWRMSLE officielle Kaggle.

## 11. Règle de sélection du champion

Le champion minimise la WAPE moyenne sur les fenêtres de validation, sous garde-fou de biais absolu. Les performances promotion/hors promotion, la stabilité entre fenêtres et le coût d'inférence sont ensuite examinés. Le holdout final ne participe jamais à cette décision.

## 12. Hiérarchie et groupes

Axe géographique :

```text
Total → state → city → store_nbr
```

Axe produit :

```text
Total → family
```

Niveau prévu : `store_nbr × family`. L'agrégation bottom-up rend les totaux cohérents par construction. MinTrace est une extension, pas un prérequis.

## 13. Simulation supply chain

La simulation compare baseline et champion avec une politique order-up-to :

- stock initial simulé ;
- lead time ;
- période de revue ;
- stock de sécurité basé sur l'erreur résiduelle ;
- coût de stockage ;
- coût de rupture ;
- niveau de service.

Les résultats sont des scénarios, pas une optimisation réelle de Favorita.

## 14. Industrialisation

- package Python réutilisable ;
- configuration YAML ;
- SQL DuckDB pour réduire la mémoire ;
- Parquet pour les données intermédiaires ;
- MLflow pour expériences et artefacts ;
- tests unitaires et anti-fuite ;
- API FastAPI de consultation des prévisions batch ;
- dashboard Streamlit ;
- Docker et CI ;
- guide GCP Cloud Run / Vertex AI.
