# Guide simple pour lancer le projet sur un PC Windows

Ce guide est destiné à une personne qui ne sait pas programmer. Il suffit de
suivre les étapes dans l'ordre et de copier-coller les commandes indiquées.

## Ce que fait l'application

L'application permet d'importer une radiographie thoracique au format JPG ou PNG.
Elle affiche ensuite :

- une estimation `NORMAL` ou `PNEUMONIA` ;
- la probabilité calculée par le modèle ;
- une carte colorée Grad-CAM montrant les zones ayant influencé le résultat.

Cette application est un prototype universitaire. Elle ne remplace pas l'avis d'un
médecin ou d'un radiologue.

## Ce qu'il faut recevoir avec le projet

Le propriétaire du projet doit fournir un dossier complet `PFE2`. Deux fichiers
indispensables doivent être présents :

```text
PFE2\models\baseline\best_model.pt
PFE2\artifacts\evaluation\baseline\evaluation.json
```

Sans ces deux fichiers, l'application ne pourra pas analyser les images.

Une radiographie JPG ou PNG peut également être fournie pour effectuer le premier
test. Le dataset complet n'est pas nécessaire pour utiliser l'application.

## Étape 1 - Vérifier le PC

Il est recommandé d'avoir :

- Windows 10 ou Windows 11 en 64 bits ;
- au moins 8 Go de mémoire RAM ;
- au moins 15 Go d'espace disque disponible ;
- une connexion Internet pour la première installation.

## Étape 2 - Installer Docker Desktop

1. Ouvrir la page officielle :
   <https://docs.docker.com/desktop/setup/install/windows-install/>
2. Télécharger **Docker Desktop for Windows**.
3. Ouvrir le fichier téléchargé `Docker Desktop Installer.exe`.
4. Conserver l'option **Use WSL 2** lorsqu'elle est proposée.
5. Terminer l'installation.
6. Redémarrer le PC si Windows le demande.
7. Ouvrir **Docker Desktop** depuis le menu Démarrer.
8. Accepter les conditions d'utilisation si elles sont affichées.
9. Attendre que Docker indique qu'il fonctionne.

Il n'est pas nécessaire de créer un compte Docker pour lancer ce projet localement.

## Étape 3 - Copier le projet sur le PC

Si le projet a été reçu sous forme de fichier ZIP :

1. Faire un clic droit sur le fichier ZIP.
2. Choisir **Extraire tout**.
3. Choisir un emplacement simple, par exemple `C:\PFE2`.
4. Ouvrir le dossier extrait.

Éviter de placer le projet dans un dossier temporaire ou dans une clé USB pendant
son exécution.

## Étape 4 - Vérifier les fichiers du modèle

Dans l'Explorateur de fichiers, vérifier que ces deux fichiers existent :

```text
C:\PFE2\models\baseline\best_model.pt
C:\PFE2\artifacts\evaluation\baseline\evaluation.json
```

Si le projet se trouve ailleurs que dans `C:\PFE2`, commencer les chemins depuis
le dossier choisi.

Si l'un des fichiers manque, demander au propriétaire du projet de le fournir avant
de continuer.

## Étape 5 - Ouvrir PowerShell dans le bon dossier

1. Ouvrir le dossier principal `PFE2` dans l'Explorateur de fichiers.
2. Cliquer dans la barre d'adresse située en haut de la fenêtre.
3. Effacer le chemin affiché.
4. Écrire `powershell`.
5. Appuyer sur la touche **Entrée**.

Une fenêtre bleue ou noire appelée PowerShell doit apparaître. Toutes les commandes
suivantes doivent être copiées dans cette fenêtre.

## Étape 6 - Créer la configuration locale

Copier cette commande, la coller dans PowerShell, puis appuyer sur **Entrée** :

```powershell
Copy-Item .env.example .env
```

Si PowerShell indique que `.env` existe déjà, ce n'est pas un problème. Ne pas
partager ce fichier, car il peut contenir des mots de passe locaux.

## Étape 7 - Lancer l'application

Vérifier que Docker Desktop est ouvert, puis copier cette commande :

```powershell
docker compose --profile application up --build -d api ui
```

La première exécution peut prendre entre 5 et 20 minutes selon la connexion Internet
et la puissance du PC. Docker télécharge Python, PyTorch et les autres composants
nécessaires. Ne pas fermer Docker Desktop pendant cette opération.

Lorsque PowerShell affiche de nouveau une ligne permettant d'écrire, copier :

```powershell
docker compose --profile application ps api ui
```

Dans la colonne `STATUS`, les services `api` et `ui` doivent finir par afficher
`healthy`. Si `starting` est affiché, attendre une minute puis relancer la même
commande.

## Étape 8 - Ouvrir l'application

Ouvrir Google Chrome, Microsoft Edge ou Firefox, puis saisir :

<http://localhost:8501>

La page **Détection de pneumonie** doit apparaître.

## Étape 9 - Tester une radiographie

1. Cliquer sur **Browse files** ou **Parcourir les fichiers**.
2. Choisir une radiographie au format JPG, JPEG ou PNG.
3. Laisser l'option Grad-CAM activée.
4. Cliquer sur **Analyser la radiographie**.
5. Attendre l'affichage du résultat.

Le résultat présente une probabilité, un seuil de décision et éventuellement une
carte colorée. Il ne doit pas être interprété comme un diagnostic médical.

## Étape 10 - Vérifier que l'API fonctionne

Dans le navigateur, ouvrir :

<http://localhost:8000/health>

Un texte similaire à celui-ci doit apparaître :

```json
{"status":"healthy","model":"baseline","threshold":0.67}
```

La documentation technique de l'API est disponible sur :

<http://localhost:8000/docs>

## Arrêter l'application

Dans PowerShell, utiliser :

```powershell
docker compose stop ui api
```

Cette commande arrête l'application sans supprimer les fichiers.

## Redémarrer l'application plus tard

1. Ouvrir Docker Desktop.
2. Ouvrir PowerShell dans le dossier `PFE2`.
3. Copier :

```powershell
docker compose start api ui
```

4. Ouvrir ensuite <http://localhost:8501>.

## Option facultative - Ouvrir la supervision

La supervision affiche le nombre de requêtes, les prédictions et la disponibilité
de l'application.

Dans PowerShell :

```powershell
docker compose --profile application --profile monitoring up -d api ui prometheus grafana
```

Ouvrir ensuite :

- Prometheus : <http://localhost:9090>
- Grafana : <http://localhost:3000>

Le nom d'utilisateur et le mot de passe Grafana sont définis dans le fichier `.env`
avec `GRAFANA_ADMIN_USER` et `GRAFANA_ADMIN_PASSWORD`.

## Problèmes fréquents

### La commande `docker` n'est pas reconnue

1. Vérifier que Docker Desktop est installé.
2. Fermer puis rouvrir PowerShell.
3. Ouvrir Docker Desktop et attendre son démarrage.
4. Réessayer la commande.

### Docker demande d'activer WSL 2

1. Faire un clic droit sur le menu Démarrer.
2. Ouvrir **Terminal (administrateur)** ou **PowerShell (administrateur)**.
3. Copier :

```powershell
wsl --update
```

4. Redémarrer le PC.
5. Relancer Docker Desktop.

### L'API indique que le modèle est introuvable

Vérifier de nouveau la présence exacte de :

```text
models\baseline\best_model.pt
artifacts\evaluation\baseline\evaluation.json
```

Les noms des dossiers et des fichiers ne doivent pas être modifiés.

### La page `localhost:8501` ne s'ouvre pas

Dans PowerShell, afficher l'état :

```powershell
docker compose --profile application ps api ui
```

Afficher ensuite les messages de l'interface :

```powershell
docker compose logs --tail 100 ui
```

### L'API ne démarre pas

Afficher ses messages :

```powershell
docker compose logs --tail 100 api
```

Rechercher notamment un message indiquant que le checkpoint ou le fichier
d'évaluation est introuvable.

### Le port 8000 ou 8501 est déjà utilisé

Ouvrir le fichier `.env` avec le Bloc-notes et remplacer :

```text
API_PORT=8000
UI_PORT=8501
```

par :

```text
API_PORT=18000
UI_PORT=18501
```

Relancer ensuite l'application. Les nouvelles adresses seront :

- interface : <http://localhost:18501>
- API : <http://localhost:18000/health>

## Obtenir de l'aide

Avant de demander de l'aide, transmettre au propriétaire du projet :

1. une capture d'écran du problème ;
2. le résultat de cette commande :

```powershell
docker compose --profile application ps api ui
```

3. les messages de l'API :

```powershell
docker compose logs --tail 100 api
```

Ne jamais envoyer le fichier `.env` ni publier les mots de passe dans une capture
d'écran.
