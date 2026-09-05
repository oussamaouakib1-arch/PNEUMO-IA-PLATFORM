# Pneumonia Detection Platform

Pour une installation destinée à un débutant complet, consulter le
[guide de démarrage pas à pas](GUIDE_DEMARRAGE_DEBUTANT.md).

Prototype académique d'aide à la décision pour détecter la pneumonie à partir de
radiographies thoraciques. Il ne remplace pas l'avis d'un professionnel de santé.

## Architecture locale

- Python 3.11 ou 3.12 pour les pipelines et modèles.
- PostgreSQL pour les métadonnées.
- MinIO comme stockage objet compatible S3.
- MLflow pour suivre les expériences et artefacts.
- PyTorch pour les modèles, ajouté via l'extra `ml`.

## Prérequis

- Python 3.11 ou 3.12
- Docker Desktop avec Docker Compose
- Git

## Installation

Sous PowerShell :

```powershell
Copy-Item .env.example .env
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Si Python 3.12 n'est pas enregistré dans le lanceur `py`, utilisez directement
le chemin de votre exécutable Python 3.12 pour créer `.venv`.

## Vérifications locales

```powershell
python -m ruff check .
python -m ruff format --check .
python -m mypy src
python -m pytest
docker compose config
```

## Services locaux

Après avoir remplacé les mots de passe de développement dans `.env` :

```powershell
docker compose up -d
docker compose ps
```

Interfaces :

- MLflow : <http://localhost:5000>
- MinIO API : <http://localhost:9000>
- MinIO Console : <http://localhost:9001>
- PostgreSQL : `localhost:5433`

Arrêt des services sans supprimer les données :

```powershell
docker compose down
```

## Structure

```text
data/            Données brutes, validées, transformées, splits et rejets
src/pneumonia/   Code Python réutilisable
tests/           Tests automatisés
notebooks/       Exploration et rapports expérimentaux
models/          Modèles locaux non versionnés
artifacts/       Artefacts locaux non versionnés
scripts/         Commandes reproductibles
```

Les radiographies, modèles et secrets ne doivent jamais être ajoutés à Git.

## Pipeline de données - phase 3

Placez le dossier Kaggle `chest_xray` dans `data/raw/chest_xray`, puis lancez :

```powershell
python -m pneumonia.data.cli --input data/raw/chest_xray
```

Le pipeline produit :

- `data/validated/catalog.csv` : métadonnées et contrôles de chaque image ;
- `data/splits/manifest.csv` : split reproductible groupé par patient ;
- `artifacts/phase3/duplicates.json` : doublons exacts SHA-256 ;
- `artifacts/phase3/data_quality_report.json` : synthèse qualité et EDA.

Les images trop petites, illisibles ou sans classe reconnue sont marquées comme rejetées.
Le split généré est stratifié approximativement par classe et garantit qu'un patient ne
figure que dans un seul ensemble.

## Baseline CNN - phase 4

Test rapide sur quelques lots :

```powershell
pneumonia-train `
  --dataset-root data/raw/kaggle/chest_xray/chest_xray `
  --epochs 1 --image-size 96 --batch-size 8 --max-batches 2 --no-mlflow
```

Pour l'entraînement suivi dans MLflow, retirez `--max-batches` et `--no-mlflow`.
Le meilleur modèle est sélectionné selon le rappel de la classe pneumonie.

## Comparaison et explicabilité - phases 5 et 6

Comparer les architectures dans un protocole commun :

```powershell
pneumonia-compare `
  --dataset-root data/raw/kaggle/chest_xray/chest_xray `
  --models baseline resnet50 densenet121 efficientnet_b0 `
  --epochs 1 --image-size 64 --batch-size 8 --max-batches 2 --no-mlflow
```

Évaluer un checkpoint, sélectionner le seuil sur la validation et produire Grad-CAM :

```powershell
pneumonia-evaluate `
  --checkpoint models/baseline/best_model.pt `
  --dataset-root data/raw/kaggle/chest_xray/chest_xray `
  --output-dir artifacts/evaluation/baseline `
  --minimum-recall 0.90
```

Le seuil est déterminé sans consulter le test. Le dossier d'évaluation contient les
métriques, la calibration, toutes les probabilités et une carte Grad-CAM.

## API et interface clinique - phase 7

Lancer l'API FastAPI :

```powershell
.\.venv\Scripts\python.exe -m uvicorn pneumonia.api.app:app --host 127.0.0.1 --port 8000
```

La documentation interactive est disponible sur <http://localhost:8000/docs> et
l'état du service sur <http://localhost:8000/health>.

Dans un second terminal, lancer l'interface Streamlit :

```powershell
.\.venv\Scripts\python.exe -m streamlit run ui/app.py
```

Ouvrez ensuite <http://localhost:8501>, importez une radiographie JPG ou PNG, puis
consultez la probabilité prédite et la carte Grad-CAM. Le résultat est une aide
expérimentale et ne constitue pas un diagnostic médical.

Dans VS Code, la tâche **Application complète** démarre l'API et l'interface en une
seule opération via **Terminal > Run Task**.

Test rapide de l'API avec une image réelle :

```powershell
.\.venv\Scripts\python.exe scripts/smoke_api.py `
  --image data/raw/kaggle/chest_xray/chest_xray/test/PNEUMONIA/person1_virus_6.jpeg
```

## Industrialisation et déploiement - phase 8

Construire et démarrer l'API et l'interface dans des conteneurs :

```powershell
docker compose --profile application up --build -d api ui
docker compose --profile application ps api ui
```

Les conteneurs s'exécutent avec un utilisateur non privilégié, sans capacités Linux,
avec contrôles de santé. Le checkpoint et le seuil d'évaluation sont montés en lecture
seule depuis l'hôte.

Métriques Prometheus : <http://localhost:8000/metrics>

Validation complète du déploiement :

```powershell
.\.venv\Scripts\python.exe scripts/validate_deployment.py `
  --image data/raw/kaggle/chest_xray/chest_xray/test/PNEUMONIA/person1_virus_6.jpeg
```

La CI GitHub vérifie Ruff, le formatage, MyPy, Bandit, les tests et les deux images
Docker. Un tag `v*` publie automatiquement les images dans GitHub Container Registry.
Le [guide d'exploitation](docs/operations.md) détaille la supervision et les commandes
de maintenance.

## Cloud, supervision et résilience - phase 9

Démarrer l'application avec Prometheus et Grafana :

```powershell
docker compose --profile application --profile monitoring up -d `
  api ui prometheus grafana
```

- Prometheus : <http://localhost:9090>
- Grafana : <http://localhost:3000>

Le tableau Grafana provisionné affiche le débit, la latence p95, les prédictions,
le taux d'erreurs et la disponibilité. Trois alertes Prometheus surveillent
l'indisponibilité, les erreurs HTTP et la latence.

Les manifests Kubernetes dans `infra/k8s/base` ajoutent des probes, limites de
ressources, autoscaling, politiques réseau, Pod Security et récupération du modèle
depuis un stockage S3. Le déploiement reste manuel tant que les identifiants du
cluster ne sont pas configurés.

Consultez le [guide de déploiement cloud](docs/cloud-deployment.md) pour préparer les
domaines, le stockage du modèle, les secrets GitHub et le workflow Kubernetes.

## Recette finale et démonstration - phase 10

Générer le rapport de recette automatisé :

```powershell
.\.venv\Scripts\python.exe scripts/final_acceptance.py
```

Livrables de clôture :

- [architecture finale](docs/architecture.md) ;
- [rapport de recette](docs/final-acceptance.md) ;
- [scénario de démonstration](docs/demo-script.md) ;
- [guide de déploiement cloud](docs/cloud-deployment.md).

Le rapport distingue explicitement les fonctionnalités validées des éléments non
couverts par le dataset, notamment les données cliniques et le NLP.
