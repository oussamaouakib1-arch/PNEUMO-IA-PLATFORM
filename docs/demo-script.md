# Scénario de démonstration

## Préparation

```powershell
docker compose --profile application --profile monitoring up -d `
  api ui prometheus grafana
docker compose --profile application --profile monitoring ps
python scripts/final_acceptance.py
```

Vérifier avant la soutenance :

- interface sur <http://localhost:8501> ;
- documentation API sur <http://localhost:8000/docs> ;
- MLflow sur <http://localhost:5000> si l'infrastructure expérimentale est lancée ;
- Prometheus sur <http://localhost:9090> ;
- Grafana sur <http://localhost:3000>.

## Démonstration de 10 minutes

1. **Problème et limite médicale - 1 minute**  
   Présenter la détection précoce et préciser immédiatement que le prototype ne
   remplace pas un radiologue.

2. **Données - 1 minute**  
   Montrer les 5 856 images, le déséquilibre des classes, les doublons détectés et
   surtout le split groupé par patient.

3. **Modèles - 2 minutes**  
   Expliquer la baseline et la comparaison avec ResNet50, DenseNet121 et
   EfficientNet-B0. Montrer le suivi MLflow.

4. **Validation - 2 minutes**  
   Présenter rappel, spécificité, ROC-AUC, PR-AUC et la sélection du seuil à 67 %.
   Insister sur les 57 faux négatifs du test.

5. **Application - 2 minutes**  
   Importer une radiographie, lancer l'analyse, commenter les probabilités et la
   carte Grad-CAM.

6. **Industrialisation - 1 minute**  
   Montrer Docker, GitHub Actions, les images GHCR et les manifests Kubernetes.

7. **Supervision et conclusion - 1 minute**  
   Montrer Grafana, les alertes, puis conclure avec les limites et les travaux futurs.

## Plan de secours

Si Docker ou le réseau échoue pendant la soutenance :

- conserver une capture de l'interface et du tableau Grafana ;
- utiliser `artifacts/evaluation/baseline/evaluation.json` pour les résultats ;
- afficher `artifacts/api_smoke/response.json` et `gradcam.png` ;
- présenter l'architecture depuis `docs/architecture.md`.
