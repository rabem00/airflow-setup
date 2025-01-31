from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import mlflow

# Set MLflow Tracking URI
MLFLOW_TRACKING_URI = "http://mlflow:5000"

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

def train_model():
    # Set tracking URI before starting MLflow run
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    with mlflow.start_run():
        # Set experiment name
        mlflow.set_experiment("minimal")
        
        mlflow.log_param("param1", 42)
        mlflow.log_metric("accuracy", 0.95)

# Define DAG
with DAG(
    '00_mlflow_minimal_training_pipeline',
    default_args=default_args,
    description='A simple ML training pipeline with MLflow integration',
    schedule_interval=None,  # Manual trigger only
    catchup=False,
) as dag:

    train_task = PythonOperator(
        task_id="train_model",
        python_callable=train_model
    )

    train_task