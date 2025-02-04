# Airflow with MLFlow integration
This repository contains a docker-compose example of Airflow with MLFlow integration. You can use it on a laptop with docker-desktop installed. It also contains some simple DAG files to test various MLFlow features.

## Setup
After cloning this repository create the following:
```bash
echo -e "AIRFLOW_UID=$(id -u)" > .env
mkdir plugins
mkdir logs
```
Run a `docker-compose build` once to build. Note: if the requirements.txt is changed then run `docker-compose build` to build the images.

After the build run:
```bash
docker-compose up airflow-init
```
## Start
Start the airflow and mlflow instances:
```bash
docker-compose up -d
```
Note: may take some minutes to start, until you see http://0.0.0.0:8080 in the log (or if started without -d on the screen). 

## UI Connections
Now go to the http://0.0.0.0:8080/ and log in using user airflow and password airflow.

Mlflow tracking server with the following configuration:
 * Tracking: local
 * Artifact Store: docker volume (local)
 * Backend: sqlite (local)

MLFlow UI: http://0.0.0.0:5000

## Create new Version with latest Airflow release
In this case get latest release 2.10.4:
https://airflow.apache.org/docs/apache-airflow/stable/release_notes.html

Get the docker-compose file (change Airflow version in URL for other docker-compose files): https://airflow.apache.org/docs/apache-airflow/2.10.4/docker-compose.yaml

Compare the docker-compose file in this repository with the downloaded one and make the nessecary changes.