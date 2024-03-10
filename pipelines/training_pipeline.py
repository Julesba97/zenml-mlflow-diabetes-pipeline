
from pathlib import Path

from zenml import pipeline

from steps.import_data import load_data
from steps.preprocess_data import prepare_data
from steps.train_model import train_model
from steps.evaluate_model import evaluate_classification_model

data_path = Path(r"C:\Users\jules\OneDrive\Documents\COURS PROJET\Mlops\ZenML\diabetes_health_indicators_project\data\diabetes_binary_health_indicators_BRFSS2015.csv")
@pipeline
def train_pipeline():
    data = load_data(path_data=data_path)
    trainset, testset = prepare_data(data)
    trained_model = train_model(trainset)
    evaluate_classification_model(trained_model, testset)