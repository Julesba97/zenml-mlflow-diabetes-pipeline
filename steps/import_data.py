import pandas as pd
import logging

from zenml import step

@step
def load_data(path_data) -> pd.DataFrame:
    """
    Charge un jeu de données à partir d'un fichier CSV.

    Args:
        path_data (str): Chemin vers le fichier CSV contenant les données.

    Returns:
        pd.DataFrame: Un DataFrame contenant les données chargées à partir du fichier CSV.
    """
    try:
        logging.info("Chargement des données en cours ...")
        data = pd.read_csv(path_data)
        logging.info("Chargement de données terminé. ")
        return data
    except Exception as e:
        logging.error(f"Une erreur s'est produite lors du chargement des données : {str(e)}")
        raise e