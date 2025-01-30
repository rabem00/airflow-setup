Get latest release:
https://airflow.apache.org/docs/apache-airflow/stable/release_notes.html

For instance 2.10.4 in this case. Get the docker-compose file (change version in URL for other docker-compose files): https://airflow.apache.org/docs/apache-airflow/2.10.4/docker-compose.yaml


```bash
echo -e "AIRFLOW_UID=$(id -u)" > .env
```

If requirements.txt is changed then run `docker-compose build` to build the images.

Setup and start:
```bash
docker-compose up airflow-init
docker-compose up -d
```

Now go to the http://localhost:8080/ and log in using user airflow and password airflow.

Mlflow tracking server with the following configuration:
 * Tracking: local
 * Artifact Store: local
 * Backend: sqlite (local)

MLFlow ui: http://localhost:5000