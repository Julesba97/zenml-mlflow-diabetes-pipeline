# Construction d'une Pipeline de Machine Learning avec ZenML et MLflow

Ce projet vise à construire une pipeline de machine learning complète pour prédire les indicateurs de 
santé liés au diabète à partir du jeu de données "Diabetes Health Indicators Dataset" trouvé sur Kaggle.

## Objectif

Le but de ce projet est de construire une pipeline de machine learning automatisée qui permettra de :

- Importer les données à partir du fichier CSV [ici](./data/diabetes_binary_health_indicators_BRFSS2015.csv)
- Prétraiter les données pour les préparer à l'entraînement du modèle.
- Entraîner un modèle de machine learning pour prédire les indicateurs de santé liés au diabète.
- Évaluer les performances du modèle entraîné.

## Structure du Projet

- **data/** : Dossier contenant les données utilisées dans le projet.
- **notebooks/** : Dossier contenant les notebooks Jupyter utilisés pour l'analyse exploratoire des données.
- **pipelines/** : Dossier contenant les définitions des pipelines ZenML pour le prétraitement, l'entraînement et l'évaluation des modèles.
- **steps/** : Dossier contenant les étapes ZenML pour importer les données, prétraiter les données, entraîner le modèle et évaluer le modèle.
- **run_pipeline.py** : Script principal de l'application qui orchestre l'exécution des pipelines ZenML et l'entraînement des modèles.

## Utilisation de ZenML
- Installez ZenML avec le support du serveur à l'aide de la commande suivante :

```bash
pip install zenml["server"]
```

- Initialisez un projet ZenML dans votre répertoire de travail en exécutant la commande suivante :

```bash
zenml init
```
- Pour démarrer un serveur ZenML sur votre machine locale, vous pouvez exécuter :

```bash
zenml up --blocking
```
## Configuration de MLflow avec ZenML

- Installez l'intégration MLflow pour ZenML en exécutant la commande suivante :
```bash
zenml integration install mlflow -y
```
- Enregistrez un tracker d'expérience MLflow avec ZenML en utilisant la commande suivante :
```bash
zenml experiment-tracker register mlflow_tracker --flavor=mlflow
```
- Enregistrez un déployeur de modèle MLflow avec ZenML en utilisant la commande suivante :
```bash
zenml model-deployer register mlflow --flavor=mlflow
```
- Enregistrez un stack MLflow avec ZenML en utilisant la commande suivante :
```bash
zenml stack register mlflow_tracker -a default -o default -d mlflow -e mlflow_tracker --set
```
## Lancer le pipeline de machine learning:
```bash
python run_pipeline.py
```
- Lancez MLflow UI en spécifiant l'emplacement du répertoire de stockage des journaux MLflow à l'aide de la commande suivante :
```bash
mlflow ui --backend-store-uri < chemin de stockage des logs de mlflow >
```



