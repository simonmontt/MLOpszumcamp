import mlflow
import joblib
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from typing import Tuple


if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter

mlflow.set_tracking_uri(uri="https://didactic-cod-wrrjgxvr69p42pgx-5000.app.github.dev/")
mlflow.set_experiment("nyc-yellow-taxi-experiment-2023-03")   


@data_exporter
def export(
    data: pd.DataFrame, 
    data_2: Tuple[DictVectorizer, LinearRegression], 
    *args, 
    **kwargs
) -> Tuple:
    """
    Log the Linear Regression model and save and log the DictVectorizer artifact using MLflow.

    Parameters:
        data (DataFrame): The DataFrame containing the dataset.
        data_2 (Tuple[DictVectorizer, LinearRegression]): A tuple containing the DictVectorizer 
            and Linear Regression model.
        model_name (str): Name to assign to the logged model in MLflow. Default is 'linear_regression'.
        artifact_name (str): Name to assign to the logged artifact in MLflow. Default is 'dict_vectorizer'.

    Returns:
        Tuple: A tuple containing the same input data.
    """
    df = data
    dv, lr = data_2 
    
    # Log the Linear Regression model
    with mlflow.start_run():
        # Log model parameters
        mlflow.log_params({
            "model_type": "Linear Regression"
        })
        # Log the model itself
        mlflow.sklearn.log_model(lr, "linear regression")
        
    # Save and log the DictVectorizer artifact
        dv_path = "artifact_" + 'dict vectorizer' + ".pkl"
        joblib.dump(dv, dv_path)
 
        mlflow.log_artifact(dv_path, 'dict vectorizer')
        
    return print("Model and Dict should be in the experiment tracking")
8