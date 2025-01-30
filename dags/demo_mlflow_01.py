from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Define default arguments
default_args = {
    'owner': 'data_scientist',
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
    context['task_instance'].xcom_push(key='X_train', value=X_train.to_dict())
    context['task_instance'].xcom_push(key='X_test', value=X_test.to_dict())
    context['task_instance'].xcom_push(key='y_train', value=y_train.to_list())
    context['task_instance'].xcom_push(key='y_test', value=y_test.to_list())

def train_model(**context):
    """Train the model and log metrics using MLflow"""
    # Get data from XCom
    ti = context['task_instance']
    X_train = pd.DataFrame(ti.xcom_pull(key='X_train'))
    y_train = ti.xcom_pull(key='y_train')
    X_test = pd.DataFrame(ti.xcom_pull(key='X_test'))
    y_test = ti.xcom_pull(key='y_test')
    
    # Start MLflow run
    with mlflow.start_run():
        # Set tracking URI - replace with your MLflow server URI
        mlflow.set_tracking_uri('http://localhost:5000')
        
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
        
        # Log model
        mlflow.sklearn.log_model(rf, "random_forest_model")

# Create DAG
dag = DAG(
    'mlflow_training_pipeline',
    default_args=default_args,
    description='A simple ML training pipeline with MLflow integration',
    schedule_interval=timedelta(days=1),
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

# Set task dependencies
preprocess_task >> train_task