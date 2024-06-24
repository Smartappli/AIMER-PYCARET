import pytest
from fastapi.testclient import TestClient

from main import (
    ModelAnomalyDetection,
    ModelClassification,
    ModelClustering,
    ModelRegression,
    ModelType,
    app,
)

client = TestClient(app)


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "pycaret_version" in response.json()


@pytest.mark.parametrize(
    "model_type, expected_enum",
    [
        (ModelType.classification, ModelClassification),
        (ModelType.regression, ModelRegression),
        (ModelType.clustering, ModelClustering),
        (ModelType.anomaly_detection, ModelAnomalyDetection),
    ],
)
def test_get_type(model_type, expected_enum):
    response = client.get(f"/model/{model_type.value}")
    assert response.status_code == 200
    enum_values = {model.name: model.value for model in expected_enum}
    assert response.json() == enum_values


def test_get_type_unknown():
    response = client.get("/model/unknown")
    assert (
        response.status_code == 422
    )  # Unprocessable Entity due to enum validation


def test_get_model_classification():
    response = client.get(f"/type/{ModelType.classification.value}")
    assert response.status_code == 200
    assert response.json() == {"model_name": ModelType.classification.value}


def test_get_model_regression():
    response = client.get(f"/type/{ModelType.regression.value}")
    assert response.status_code == 200
    assert response.json() == {"model_name": ModelType.regression.value}


def test_get_model_clustering():
    response = client.get(f"/type/{ModelType.clustering.value}")
    assert response.status_code == 200
    assert response.json() == {"model_name": ModelType.clustering.value}


def test_get_model_anomaly_detection():
    response = client.get(f"/type/{ModelType.anomaly_detection.value}")
    assert response.status_code == 200
    assert response.json() == {"model_name": ModelType.anomaly_detection.value}

def test_train_anomaly_detection_model():
    params = AnomalyDetectionParams(model="iforest", fraction=0.1, verbose=True)
    response = client.post("/anomaly-detection", json=params.dict())
    assert response.status_code == 200
    assert response.json()["model_type"] == "anomaly_detection"
    assert response.json()["params"]["model"] == "iforest"

def test_train_classification_model():
    params = ClassificationParams(estimator="rf", fold=5, round=3, cross_validation=True)
    response = client.post("/classification", json=params.dict())
    assert response.status_code == 200
    assert response.json()["model_type"] == "classification"
    assert response.json()["params"]["estimator"] == "rf"

def test_train_clustering_model():
    params = ClusteringParams(model="kmeans", num_clusters=3, verbose=True)
    response = client.post("/clustering", json=params.dict())
    assert response.status_code == 200
    assert response.json()["model_type"] == "clustering"
    assert response.json()["params"]["model"] == "kmeans"

def test_train_regression_model():
    params = RegressionParams(estimator="lr", fold=10, round=2, cross_validation=True)
    response = client.post("/regression", json=params.dict())
    assert response.status_code == 200
    assert response.json()["model_type"] == "regression"
    assert response.json()["params"]["estimator"] == "lr"

def test_train_time_series_model():
    params = TimeSeriesParams(estimator="arima", fold=3, round=4, cross_validation=True)
    response = client.post("/time-series", json=params.dict())
    assert response.status_code == 200
    assert response.json()["model_type"] == "time_series"
    assert response.json()["params"]["estimator"] == "arima"
