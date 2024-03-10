import pandas as pd
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score,
                             roc_auc_score, log_loss)
from sklearn.base import ClassifierMixin
import logging
from typing import Dict
import mlflow

from zenml import step
from zenml.client import Client

experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def evaluate_classification_model(model:  ClassifierMixin, testset: pd.DataFrame) -> Dict[str, float]:
    """
    Évalue un modèle de classification à l'aide de différentes métriques de performance.

    Args:
        model : Modèle scikit-learn entraîné.
        X_test : Caractéristiques (features) de l'ensemble de données de test.
        y_test : Étiquettes (labels) correspondantes de l'ensemble de données de test.

    Returns:
        dict : Dictionnaire contenant les métriques évaluées.
    """
    try:
        logging.info("Évaluation du modèle de classification en cours...")
        
        target_name = "Diabetes_binary"
        X_test = testset.drop(columns=[target_name], axis=1)
        y_test = testset[target_name]
        y_pred = model.predict(X_test)
    
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred)
        logloss = log_loss(y_test, y_pred)
        
        logging.info("Évaluation du modèle de classification terminée.")
    
        
        
        
        metrics = {
            'Accuracy': round(accuracy, 3),
            'Precision': round(precision, 3),
            'Recall': round(recall, 3),
            'F1 Score': round(f1, 3),
            'ROC AUC': round(roc_auc, 3),
            'Log Loss': round(logloss, 3)
        }
        
        logging.info("Métriques évaluées :")
        for metric, value in metrics.items():
            logging.info(f"{metric}: {value}")
            mlflow.log_metric(metric, value)
        
        return metrics
    except Exception as e:
        logging.error(f"Une erreur s'est produite lors de l'évaluation du modèle : {str(e)}")
        raise e
