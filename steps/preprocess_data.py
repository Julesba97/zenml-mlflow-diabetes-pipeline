import pandas as pd
import logging
from typing_extensions import Annotated
from typing import Tuple
from sklearn.model_selection import train_test_split

from zenml import step

@step
def prepare_data(data: pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame, "trainset"], 
    Annotated[pd.DataFrame, "testset"]
]:
    """
    Prépare les données en les divisant en ensembles de données d'entraînement et de test.

    Args:
        data : DataFrame pandas contenant les données à diviser.

    Returns:
        Tuple contenant les ensembles de données d'entraînement et de test.
        trainset : Ensemble de données d'entraînement.
        testset : Ensemble de données de test.
    """
    try:
        logging.info("Préparation des données en cours...")
        no_duplicated_data = data[data.duplicated(keep="first") == False]
        trainset, testset = train_test_split(no_duplicated_data, test_size=0.2, random_state=42)
        logging.info("Calcul des dimensions de trainset et testset...")
        trainset_rows, trainset_cols = trainset.shape
        testset_rows, testset_cols = testset.shape
        logging.info(f"Nombre de lignes et de colonnes de trainset : {trainset_rows} lignes, {trainset_cols} colonnes")
        logging.info(f"Nombre de lignes et de colonnes de testset : {testset_rows} lignes, {testset_cols} colonnes")

        logging.info("Préparation des données terminée.")
        return trainset, testset
    except Exception as e:
        logging.error(f"Une erreur s'est produite lors de la préparation des données : {str(e)}")
        raise e

