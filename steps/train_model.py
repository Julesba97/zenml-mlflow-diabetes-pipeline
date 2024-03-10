
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
import logging

import mlflow

from zenml import step
from zenml.client import Client

experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def train_model(trainset: pd.DataFrame) -> ClassifierMixin:
    """
    Entraîne un modèle avec des données d'entraînement.

    Args:
        trainset : Caractéristiques (features) des données d'entraînement Et
               Étiquettes (labels) correspondantes des données d'entraînement.

    Returns:
        Le modèle entraîné.
    """
    try:
        target_name = "Diabetes_binary"
        X_train = trainset.drop(columns=[target_name], axis=1)
        y_train = trainset[target_name]
        mlflow.sklearn.autolog()
        logging.info("Entraînement du modèle en cours...")
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        
        logging.info("Entraînement du modèle terminé.")
        
        return model
    except Exception as e:
        logging.error(f"Une erreur s'est produite lors de l'entraînement du modèle : {str(e)}")
        raise e
