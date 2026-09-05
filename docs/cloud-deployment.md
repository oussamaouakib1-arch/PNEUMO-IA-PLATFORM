# Déploiement Kubernetes

Cette base est indépendante du fournisseur cloud. Elle suppose un cluster Kubernetes
avec un contrôleur Ingress NGINX, Metrics Server et un accès à un stockage compatible S3.

## 1. Publier les images

Créer un tag versionné déclenche la publication dans GHCR :

```powershell
git tag v0.1.0
git push origin v0.1.0
```

Les images attendues sont :

- `ghcr.io/oussamaouakib1-arch/PNEUMO-IA-PLATFORM/pneumonia-api:0.1.0`
- `ghcr.io/oussamaouakib1-arch/PNEUMO-IA-PLATFORM/pneumonia-ui:0.1.0`

## 2. Déposer le modèle dans le stockage objet

Copier les deux fichiers suivants dans le bucket configuré :

- `models/baseline/best_model.pt` vers `baseline/best_model.pt`
- `artifacts/evaluation/baseline/evaluation.json` vers `baseline/evaluation.json`

Le conteneur d'initialisation télécharge ces fichiers au démarrage de chaque Pod API.
Les identifiants ne sont jamais stockés dans Git.

## 3. Adapter la configuration

Modifier `infra/k8s/base/configmap.yaml` :

- `MODEL_S3_ENDPOINT` et `MODEL_S3_BUCKET` ;
- les chemins des deux objets si nécessaire.

Modifier `infra/k8s/base/ingress.yaml` :

- les noms de domaine de l'interface et de l'API ;
- le nom du secret TLS ;
- `ingressClassName` si le cluster n'utilise pas NGINX.

Les politiques réseau autorisent les namespaces `ingress-nginx` et `monitoring`.
Adapter leurs sélecteurs si le fournisseur utilise d'autres noms.

## 4. Valider sans déployer

```powershell
kubectl kustomize infra/k8s/base |
  docker run --rm -i ghcr.io/yannh/kubeconform:v0.7.0 -strict -summary
```

## 5. Déploiement manuel

Créer le secret du stockage puis appliquer les manifests :

```powershell
kubectl apply -f infra/k8s/base/namespace.yaml
kubectl -n pneumonia create secret generic model-storage `
  --from-literal=access-key="VOTRE_CLE" `
  --from-literal=secret-key="VOTRE_SECRET"
kubectl apply -k infra/k8s/base
kubectl -n pneumonia rollout status deployment/pneumonia-api --timeout=10m
kubectl -n pneumonia rollout status deployment/pneumonia-ui --timeout=5m
```

## 6. Déploiement GitHub Actions

Le workflow `deploy-kubernetes.yml` est volontairement manuel. Configurer
l'environnement GitHub `production` avec :

- `KUBE_CONFIG_BASE64` : kubeconfig encodé en base64 ;
- `MODEL_S3_ACCESS_KEY` ;
- `MODEL_S3_SECRET_KEY`.

Lancer ensuite **Deploy to Kubernetes** dans GitHub Actions et fournir le tag de
l'image. Les règles d'approbation de l'environnement GitHub peuvent être utilisées
pour exiger une validation humaine avant chaque déploiement.

## 7. Contrôles après déploiement

```powershell
kubectl -n pneumonia get pods,services,ingress,hpa
kubectl -n pneumonia logs deployment/pneumonia-api --tail=100
kubectl -n pneumonia port-forward service/pneumonia-api 8000:8000
python scripts/validate_deployment.py `
  --image data/raw/kaggle/chest_xray/chest_xray/test/PNEUMONIA/person1_virus_6.jpeg
```

Le système reste un prototype académique et ne doit pas être utilisé pour établir
un diagnostic médical réel.
