# Guide d'exploitation

## Démarrage conteneurisé

Le checkpoint `models/baseline/best_model.pt` et le rapport
`artifacts/evaluation/baseline/evaluation.json` doivent exister sur l'hôte.
Ils sont montés en lecture seule dans le conteneur API et ne sont jamais intégrés
à l'image.

```powershell
Copy-Item .env.example .env
docker compose --profile application up --build -d api ui
docker compose --profile application ps api ui
```

Services :

- API : <http://localhost:8000>
- Documentation OpenAPI : <http://localhost:8000/docs>
- Interface : <http://localhost:8501>
- Métriques Prometheus : <http://localhost:8000/metrics>

Les ports hôtes peuvent être modifiés dans `.env` avec `API_PORT` et `UI_PORT`.

## Validation

```powershell
python scripts/validate_deployment.py `
  --image data/raw/kaggle/chest_xray/chest_xray/test/PNEUMONIA/person1_virus_6.jpeg
```

Le script vérifie l'API, l'interface, une inférence réelle et la présence des
métriques de prédiction.

## Supervision

Les conteneurs exposent des contrôles de santé Docker. L'API publie notamment :

- `pneumonia_http_requests_total` par méthode, chemin et statut ;
- `pneumonia_http_request_duration_seconds` pour la latence ;
- `pneumonia_predictions_total` par modèle et résultat.

Consulter l'état et les journaux :

```powershell
docker compose --profile application ps
docker compose logs --tail 100 api ui
```

Démarrer la collecte Prometheus et le tableau Grafana :

```powershell
docker compose --profile application --profile monitoring up -d `
  api ui prometheus grafana
```

Interfaces supplémentaires :

- Prometheus : <http://localhost:9090>
- Grafana : <http://localhost:3000>

Les identifiants Grafana sont définis avec `GRAFANA_ADMIN_USER` et
`GRAFANA_ADMIN_PASSWORD` dans `.env`. Le tableau **Pneumonia API - Exploitation**
est provisionné automatiquement.

Test de charge léger :

```powershell
python scripts/load_test.py `
  --image data/raw/kaggle/chest_xray/chest_xray/test/PNEUMONIA/person1_virus_6.jpeg `
  --requests 10 --concurrency 2
```

## Arrêt et redémarrage

```powershell
docker compose stop api ui
docker compose start api ui
```

Pour reconstruire après une modification du code :

```powershell
docker compose --profile application up --build -d --force-recreate api ui
```

## Publication

Le workflow `.github/workflows/ci.yml` contrôle la qualité, la sécurité, les tests
et la construction des images à chaque pull request ou push sur `main`/`develop`.

Un tag Git au format `v1.2.3` déclenche `.github/workflows/release.yml`, qui publie
les images API et UI dans GitHub Container Registry avec le jeton GitHub du dépôt.
