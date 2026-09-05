# Architecture finale

```mermaid
flowchart LR
    D["Radiographies Kaggle"] --> Q["Contrôle qualité et split patient"]
    Q --> T["Entraînement CNN et comparaison"]
    T --> M["MLflow, MinIO et PostgreSQL"]
    T --> E["Évaluation indépendante et calibration"]
    E --> C["Checkpoint et seuil validé"]
    C --> A["API FastAPI"]
    A --> U["Interface Streamlit"]
    A --> P["Métriques Prometheus"]
    P --> G["Tableau Grafana et alertes"]

    R["GitHub Actions"] --> I["Images GHCR avec SBOM"]
    I --> K["Déploiement Kubernetes"]
    S["Stockage S3 du modèle"] --> K
    K --> A
    K --> U
```

Le pipeline de données sépare les patients avant l'entraînement afin de limiter les
fuites entre apprentissage, validation et test. Le seuil de décision est choisi sur
la validation, puis figé avant l'évaluation finale.

Les images Docker ne contiennent ni données médicales ni modèle. En production,
un conteneur d'initialisation Kubernetes récupère le checkpoint et son seuil depuis
un stockage objet protégé.
