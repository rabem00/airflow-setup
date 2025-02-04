from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import mlflow

# Set MLflow Tracking URI
MLFLOW_TRACKING_URI = "http://mlflow:5000"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Default args for the DAG
default_args = {
    "owner": "Marco Rabelink",
    "depends_on_past": False,
    "start_date": datetime(2025, 1, 30),
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

def list_registered_models():
    """List all registered models in MLflow"""
    client = mlflow.MlflowClient()
    models = client.search_registered_models()

    for model in models:
        print(f"Model: {model.name}, Latest Version: {model.latest_versions}")

def get_model_details():
    """Retrieve details of a specific model"""
    client = mlflow.MlflowClient()
    model_name = "random_forest_model"

    # Search for all versions of the model
    model_versions = client.search_model_versions(f"name='{model_name}'")

    for mv in model_versions:
        print(f"Model: {mv.name}, Version: {mv.version}, Source: {mv.source}, Status: {mv.status}")

# Define DAG
dag = DAG(
    "02_mlflow_model_retrieval",
    default_args=default_args,
    description="DAG to list registered models and retrieve model details from MLflow",
    schedule_interval=None,  # Manual trigger only
    catchup=False,
)

# Define Tasks
list_models_task = PythonOperator(
    task_id="list_registered_models",
    python_callable=list_registered_models,
    dag=dag,
)

get_model_details_task = PythonOperator(
    task_id="get_model_details",
    python_callable=get_model_details,
    dag=dag,
)

# Task Dependencies
list_models_task >> get_model_details_task