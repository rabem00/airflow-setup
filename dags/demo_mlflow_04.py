from airflow import DAG
from datetime import datetime, timedelta
import mlflow
import mlflow.sklearn
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from airflow.operators.python import PythonOperator

# Custom operator to suppress logging of the return value.
class NoReturnPythonOperator(PythonOperator):
    def execute(self, context):
        # Execute the callable but do not push the return value to XCom.
        result = self.python_callable(*self.op_args, **self.op_kwargs)
        return None  # Always return None without logging it.

# Set MLflow Tracking URI
mlflow.set_tracking_uri("http://mlflow:5000")

# Define default arguments
default_args = {
    'owner': 'Marco Rabelink',
    'depends_on_past': False,
    'start_date': datetime(2025, 1, 30),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

# Create a logger for the DAG tasks
logger = logging.getLogger("airflow.task")

def train_model():
    # Load sample dataset
    df = pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv")
    X = df.drop(columns=["species"])
    y = df["species"].astype("category").cat.codes

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    with mlflow.start_run() as run:
        # Log parameters
        n_estimators = 100
        max_depth = 5
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)

        # Train model
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        model.fit(X_train, y_train)

        # Evaluate model
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        # Log metrics
        mlflow.log_metric("accuracy", acc)

        # Save and log model artifact
        model_path = "random_forest_model.pkl"
        joblib.dump(model, model_path)
        mlflow.sklearn.log_model(model, "model")
        logger.info(f"Model logged with accuracy: {acc}")

        # Generate and log confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(5, 5))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.colorbar()
        plt.xlabel("Predicted label")
        plt.ylabel("True label")
        plt.savefig("confusion_matrix.png")
        mlflow.log_artifact("confusion_matrix.png")

def register_model():
    client = mlflow.tracking.MlflowClient()
    # Retrieve the run with the highest accuracy from the experiment with id "0"
    run_info = client.search_runs(experiment_ids=["0"], order_by=["metrics.accuracy DESC"])[0].info
    run_id = run_info.run_id
    # Register the model from the run's artifact location.
    result = mlflow.register_model(f"runs:/{run_id}/model", "IrisClassifier")
    logger.info(f"Model registered: {result.name} (version {result.version})")

def run_inference():
    # Load the registered model. Here, we assume version 1.
    model_uri = "models:/IrisClassifier/1"
    model = mlflow.sklearn.load_model(model_uri)
    
    # Load the iris dataset and prepare a sample input.
    df = pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv")
    sample = df.drop(columns=["species"]).iloc[0:1]
    
    # Get the prediction from the model
    pred_code = model.predict(sample)[0]
    
    # Map the numeric prediction back to species names.
    species_mapping = dict(enumerate(sorted(df["species"].unique())))
    predicted_species = species_mapping[pred_code]
    
    # Create a visually appealing output message.
    message = (
        "\n============================================\n"
        "        Inference Prediction Result\n"
        "============================================\n"
        f"The model predicts that the sample is: {predicted_species.upper()}\n"
        "============================================\n"
    )
    logger.info(message)

# Define the DAG
with DAG(
    dag_id="04_mlflow_register_and_infer",
    default_args=default_args,
    description='A training pipeline with MLflow integration, model registration, and inference',
    schedule_interval=None,  # Manual trigger only
    catchup=False
) as dag:
    
    train_task = NoReturnPythonOperator(
        task_id="train_model",
        python_callable=train_model
    )
    
    register_task = NoReturnPythonOperator(
        task_id="register_model",
        python_callable=register_model
    )
    
    inference_task = NoReturnPythonOperator(
        task_id="run_inference",
        python_callable=run_inference
    )
    
    # Set task dependencies: first train, then register, then run inference.
    train_task >> register_task >> inference_task