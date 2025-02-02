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
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# MLflow Tracking URI
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

def train_model():
    # Load sample dataset
    df = pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv")
    X = df.drop(columns=["species"])
    y = df["species"].astype("category").cat.codes
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    with mlflow.start_run():
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
        
        # Save and log model
        model_path = "random_forest_model.pkl"
        joblib.dump(model, model_path)
        mlflow.sklearn.log_model(model, "model")
        print(f"Model logged with accuracy: {acc}")

        # Generate and log confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(5,5))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.colorbar()
        plt.xlabel("Predicted label")
        plt.ylabel("True label")
        plt.savefig("confusion_matrix.png")
        mlflow.log_artifact("confusion_matrix.png")


def register_model():
    client = mlflow.tracking.MlflowClient()
    run_id = client.search_runs(experiment_ids=["0"], order_by=["metrics.accuracy DESC"])[0].info.run_id
    mlflow.register_model(f"runs:/{run_id}/model", "IrisClassifier")

# Define DAG
with DAG(
    dag_id="03_mlflow_register_model",
    default_args=default_args,
    description='A training pipeline with MLflow integration - register model',
    schedule_interval=None,  # Manual trigger only
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
    
    train_task >> register_task
