# Exigences métier et critères d'acceptation

## Objectif

Améliorer la précision des ventes à 16 jours par rapport aux baselines saisonnières, sans introduire de sous-prévision systématique.

## Utilisateurs cibles

- demand planner ;
- responsable magasin ;
- analyste promotion ;
- équipe Data & AI ;
- interlocuteur supply chain.

## Critères d'acceptation

1. Le modèle champion doit battre la meilleure baseline sur la WAPE moyenne des validations.
2. Le biais absolu moyen doit respecter le garde-fou configuré ou être explicitement justifié.
3. Le holdout final ne doit jamais intervenir dans le choix du modèle.
4. Les features de ventes doivent être strictement causales.
5. Les prévisions bottom-up doivent être cohérentes aux niveaux magasin, ville, state et famille.
6. Les résultats doivent être disponibles par magasin, famille, promotion et horizon.
7. La simulation de stock doit distinguer clairement données réelles et hypothèses.
8. Le pipeline doit être réexécutable par configuration et couvert par des tests.
