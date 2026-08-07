# Architecture GCP proposée

## Flux batch recommandé

1. Fichiers bruts dans Cloud Storage.
2. Job de préparation dans Vertex AI Custom Job ou Cloud Run Job.
3. Données curées et features dans Cloud Storage en Parquet ou BigQuery.
4. Entraînement et suivi dans Vertex AI Experiments / MLflow.
5. Prévisions batch écrites dans BigQuery.
6. API FastAPI déployée sur Cloud Run pour consulter les prévisions.
7. Dashboard Streamlit sur Cloud Run ou Looker Studio sur BigQuery.
8. Cloud Scheduler déclenche les jobs quotidiens.
9. Cloud Monitoring surveille erreurs, latence, dérive et fraîcheur.

## Sécurité et conformité

- comptes de service à privilèges minimaux ;
- secrets dans Secret Manager ;
- images dans Artifact Registry ;
- logs structurés ;
- séparation environnements dev, staging et production ;
- artefacts et données versionnés.

Les fichiers `Dockerfile`, `deploy/cloudbuild.yaml` et `deploy/cloudrun-service.yaml` servent de base. Les identifiants GCP restent à renseigner par l'utilisateur.
