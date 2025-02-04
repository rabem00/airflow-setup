from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

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

def load_data():
    """Load sample data for demonstration"""
    from sklearn.datasets import load_iris
    iris = load_iris()
    data = pd.DataFrame(iris.data, columns=iris.feature_names)
    data['target'] = iris.target
    return data

def preprocess_data(**context):
    """Preprocess the data and split into train/test sets"""
    data = load_data()
    X = data.drop('target', axis=1)
    y = data['target']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Push to XCom for next tasks
    context['task_instance'].xcom_push(key='X_train', value=X_train.to_json())
    context['task_instance'].xcom_push(key='X_test', value=X_test.to_json())
    context['task_instance'].xcom_push(key='y_train', value=y_train.tolist())
    context['task_instance'].xcom_push(key='y_test', value=y_test.tolist())

def train_model(**context):
    """Train the model and log metrics using MLflow"""
    # Get data from XCom
    ti = context['task_instance']
    X_train = pd.read_json(ti.xcom_pull(key='X_train'))
    y_train = ti.xcom_pull(key='y_train')
    X_test = pd.read_json(ti.xcom_pull(key='X_test'))
    y_test = ti.xcom_pull(key='y_test')
    
    # Set tracking URI before starting MLflow run
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    #mlflow.tracking.MlflowClient().delete_experiment("2")

    # Ensure the experiment exists and retrieve its ID
    experiment_name = "random_forest_model"
    experiment = mlflow.get_experiment_by_name(experiment_name)

    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
    else:
        experiment_id = experiment.experiment_id

    # Start MLflow run with the correct experiment
    with mlflow.start_run(experiment_id=experiment_id):
        # Train model
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        
        # Make predictions
        y_pred = rf.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        
        # Log parameters
        mlflow.log_param('n_estimators', 100)
        
        # Log metrics
        mlflow.log_metrics({
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall
        })
        
        # Define an input example (first row of X_train)
        input_example = X_train.iloc[:1]

        # Log model with input example to fix warning
        mlflow.sklearn.log_model(
            rf,
            "random_forest_model",
            input_example=input_example
        )

def register_model():
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name("random_forest_model")
    run_id = client.search_runs(experiment_ids=[experiment.experiment_id], order_by=["metrics.accuracy DESC"])[0].info.run_id
    mlflow.register_model(f"runs:/{run_id}/model", "random_forest_model")

# Create DAG
dag = DAG(
    '01_mlflow_training_pipeline',
    default_args=default_args,
    description='A ML training pipeline with MLflow integration - random_forest_model',
    schedule_interval=None,  # Manual trigger only
    catchup=False
)

# Define tasks
preprocess_task = PythonOperator(
    task_id='preprocess_data',
    python_callable=preprocess_data,
    provide_context=True,
    dag=dag
)

train_task = PythonOperator(
    task_id='train_model',
    python_callable=train_model,
    provide_context=True,
    dag=dag
)

register_task = PythonOperator(
        task_id="register_model",
        python_callable=register_model
)

# Set task dependencies
preprocess_task >> train_task >> register_task