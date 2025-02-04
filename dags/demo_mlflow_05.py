from airflow import DAG
from airflow.operators.python import PythonOperator
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
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Set MLflow Tracking URI
mlflow.set_tracking_uri("http://mlflow:5000")

# Define default arguments for the DAG
default_args = {
    'owner': 'Your Name',
    'depends_on_past': False,
    'start_date': datetime(2025, 1, 30),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

# Create a logger
logger = logging.getLogger("airflow.task")

def train_model():
    """
    Simulate loading city-bike trip data, train a regression model to predict trip duration
    from trip distance, and log the model and metrics with MLflow.
    """
    # Simulate city-bike data
    # For demonstration, we create 1000 samples with:
    # - trip_distance (in km) uniformly distributed between 0.5 and 5 km
    # - trip_duration (in minutes) roughly proportional to distance plus noise
    np.random.seed(42)
    n_samples = 1000
    trip_distance = np.random.uniform(0.5, 5.0, n_samples)
    noise = np.random.normal(0, 3, n_samples)
    # Assume an average speed such that duration (min) = 12 * distance (km) + noise
    trip_duration = 12 * trip_distance + noise

    df = pd.DataFrame({
        "trip_distance": trip_distance,
        "trip_duration": trip_duration
    })

    # Split into train and test sets
    X = df[["trip_distance"]]
    y = df["trip_duration"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    with mlflow.start_run() as run:
        # Set model parameters
        n_estimators = 100
        max_depth = 5

        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)

        # Train a regression model
        model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        model.fit(X_train, y_train)

        # Evaluate the model using RMSE
        predictions = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        mlflow.log_metric("rmse", rmse)
        logger.info(f"Model trained. RMSE on test set: {rmse:.2f}")

        # Log the model as an MLflow artifact
        model_path = "citybike_model.pkl"
        joblib.dump(model, model_path)
        mlflow.sklearn.log_model(model, "model")
        logger.info("Model logged with MLflow.")

def register_model():
    """
    Register the best model from the MLflow experiment.
    """
    client = mlflow.tracking.MlflowClient()
    # Retrieve the best run from experiment id "0" based on RMSE (lowest is better)
    runs = client.search_runs(experiment_ids=["0"], order_by=["metrics.rmse ASC"])
    best_run = runs[0]
    run_id = best_run.info.run_id
    result = mlflow.register_model(f"runs:/{run_id}/model", "CityBikeDurationPredictor")
    logger.info(f"Model registered: {result.name} (version {result.version})")

def run_inference():
    """
    Load the registered model and perform inference on a sample city-bike trip.
    An annotated image is generated to display the input trip distance and predicted trip duration.
    """
    # Load the registered model (assuming version 1 for simplicity)
    model_uri = "models:/CityBikeDurationPredictor/1"
    model = mlflow.sklearn.load_model(model_uri)

    # Define a sample input (e.g., a trip of 2.5 km)
    sample_distance = 2.5
    predicted_duration = model.predict([[sample_distance]])[0]

    # Create a custom annotated plot with the inference result
    plt.figure(figsize=(6, 4))
    plt.text(0.5, 0.7, f"Trip Distance: {sample_distance:.1f} km", fontsize=16,
             ha='center', transform=plt.gca().transAxes)
    plt.text(0.5, 0.5, f"Predicted Duration: {predicted_duration:.1f} minutes", fontsize=16,
             ha='center', transform=plt.gca().transAxes)
    plt.text(0.5, 0.3, "City-Bike Trip Inference", fontsize=18, ha='center',
             transform=plt.gca().transAxes, color="navy")
    plt.axis("off")
    
    # Save the inference result as an image
    inference_image = "citybike_inference.png"
    plt.savefig(inference_image, bbox_inches='tight')
    plt.close()
    
    # Log the inference image as an MLflow artifact
    mlflow.log_artifact(inference_image)
    logger.info(f"Inference completed. See artifact: {inference_image}")

# Define the DAG
with DAG(
    dag_id="05_mlflow_citybike",
    default_args=default_args,
    description="City-bike training and inference pipeline with MLflow",
    schedule_interval=None,  # Trigger manually
    catchup=False
) as dag:
    
    train_task = PythonOperator(
        task_id="train_model",
        python_callable=train_model
    )
    
    register_task = PythonOperator(
        task_id="register_model",
        python_callable=register_model
    )
    
    inference_task = PythonOperator(
        task_id="run_inference",
        python_callable=run_inference,
        do_xcom_push=False
    )
    
    # Define task dependencies
    train_task >> register_task >> inference_task