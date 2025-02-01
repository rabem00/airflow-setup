from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import mlflow
import mlflow.sklearn
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error as mse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO

BASE_URL = "https://metaflow-demo-public.s3.us-west-2.amazonaws.com/taxi/clean"
TRAIN_URL = BASE_URL + "/train_sample.parquet"
TEST_URL = BASE_URL + "/test.parquet"
FEATURES = [
    "pickup_year",
    "pickup_dow",
    "pickup_hour",
    "abs_distance",
    "pickup_longitude",
    "dropoff_longitude",
]

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

MLFLOW_TRACKING_URI = "http://mlflow:5000"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# If the mlflow.set_experiment is used then during the dag import
# the experiment is already created. But if someone removes the experiment
# the dag gives an import error.
# mlflow.set_experiment("fare_regression_model")
# The following is a better solution.
experiment_name = "fare_regression_model"
experiment = mlflow.get_experiment_by_name(experiment_name)

if experiment is None:
    experiment_id = mlflow.create_experiment(experiment_name)
else:
    experiment_id = experiment.experiment_id

def make_random_forest():
    """Builds a Random Forest model pipeline."""
    ct_pipe = ColumnTransformer(
        transformers=[
            ("encoder", OneHotEncoder(categories="auto"), ["pickup_dow"]),
            (
                "std_scaler",
                StandardScaler(),
                ["abs_distance", "pickup_longitude", "dropoff_longitude"],
            ),
        ]
    )
    return Pipeline(
        [
            ("ct", ct_pipe),
            (
                "forest_reg",
                RandomForestRegressor(
                    n_estimators=10, n_jobs=-1, random_state=3, max_features=8
                ),
            ),
        ]
    )


def plot(correct, predicted):
    """Generates a scatter plot of actual vs. predicted fares."""
    MAX_FARE = 100
    line = np.arange(0, MAX_FARE, MAX_FARE / 1000)
    plt.rcParams.update({"font.size": 22})
    plt.scatter(x=correct, y=predicted, alpha=0.01, linewidth=0.5)
    plt.plot(line, line, linewidth=2, color="black")
    plt.xlabel("Correct fare")
    plt.ylabel("Predicted fare")
    plt.xlim([0, MAX_FARE])
    plt.ylim([0, MAX_FARE])
    fig = plt.gcf()
    fig.set_size_inches(18, 10)
    buf = BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    return buf


def start(**kwargs):
    """Loads training data and stores it in XCom."""
    train_df = pd.read_parquet(TRAIN_URL)
    kwargs["ti"].xcom_push("X_train", train_df[FEATURES].to_dict())
    kwargs["ti"].xcom_push("y_train", train_df["fare_amount"].tolist())


def model(**kwargs):
    """Trains the model and logs it to MLflow."""
    ti = kwargs["ti"]
    X_train = pd.DataFrame(ti.xcom_pull(task_ids="start", key="X_train"))
    y_train = ti.xcom_pull(task_ids="start", key="y_train")

    rf_model = make_random_forest()

    # Start MLflow run with the correct experiment
    with mlflow.start_run(experiment_id=experiment_id): 
        rf_model.fit(X_train, y_train)

        # Log parameters
        mlflow.log_param("n_estimators", 10)
        mlflow.log_param("max_features", 8)
        mlflow.log_param("random_state", 3)

        # Log model
        mlflow.sklearn.log_model(rf_model, "random_forest_model")

    ti.xcom_push("rf_model", rf_model)


def evaluate(**kwargs):
    """Evaluates the model and logs metrics to MLflow."""
    ti = kwargs["ti"]
    rf_model = ti.xcom_pull(task_ids="model", key="rf_model")
    test_df = pd.read_parquet(TEST_URL)

    X_test = test_df[FEATURES]
    y_test = test_df["fare_amount"]
    y_pred = rf_model.predict(X_test)

    # Baseline prediction (mean of y_test)
    y_baseline_pred = np.repeat(y_test.mean(), y_test.shape[0])

    # Compute RMSE
    model_rmse = mse(y_test, y_pred, squared=False)
    baseline_rmse = mse(y_test, y_baseline_pred, squared=False)

    # Start MLflow run with the correct experiment
    with mlflow.start_run(experiment_id=experiment_id):
        mlflow.log_metric("model_rmse", model_rmse)
        mlflow.log_metric("baseline_rmse", baseline_rmse)

    ti.xcom_push("model_rmse", model_rmse)
    ti.xcom_push("baseline_rmse", baseline_rmse)
    ti.xcom_push("y_test", y_test.tolist())
    ti.xcom_push("y_pred", y_pred.tolist())


def create_report(**kwargs):
    """Generates a report comparing model and baseline performance."""
    ti = kwargs["ti"]
    y_test = ti.xcom_pull(task_ids="evaluate", key="y_test")
    y_pred = ti.xcom_pull(task_ids="evaluate", key="y_pred")
    model_rmse = ti.xcom_pull(task_ids="evaluate", key="model_rmse")
    baseline_rmse = ti.xcom_pull(task_ids="evaluate", key="baseline_rmse")

    # Generate scatter plot
    plot_image = plot(y_test, y_pred)

    # Log the plot to MLflow
    with mlflow.start_run(experiment_id=experiment_id):
        mlflow.log_artifact(plot_image, artifact_path="plots")

    print("# Model Report")
    print(f"Random Forest RMSE: {model_rmse}")
    print(f"Baseline RMSE: {baseline_rmse}")


with DAG(
    "03_mlflow_training_pipeline",
    default_args=default_args,
    description='A ML training pipeline with MLflow - fare_regression_flow',
    schedule_interval=None,
    catchup=False,
) as dag:
    start_task = PythonOperator(task_id="start", python_callable=start)
    model_task = PythonOperator(task_id="model", python_callable=model)
    evaluate_task = PythonOperator(task_id="evaluate", python_callable=evaluate)
    report_task = PythonOperator(task_id="create_report", python_callable=create_report)

    start_task >> model_task >> evaluate_task >> report_task