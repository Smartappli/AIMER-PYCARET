import pytest
from fastapi.testclient import TestClient
from main import app, ModelType, ModelClassification, ModelRegression, ModelClustering, ModelAnomalyDetection

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "pycaret_version" in response.json()

@pytest.mark.parametrize("model_type, expected_enum", [
    (ModelType.classification, ModelClassification),
    (ModelType.regression, ModelRegression),
    (ModelType.clustering, ModelClustering),
    (ModelType.anomaly_detection, ModelAnomalyDetection),
])
def test_get_type(model_type, expected_enum):
    response = client.get(f"/model/{model_type.value}")
    assert response.status_code == 200
    enum_values = {model.name: model.value for model in expected_enum}
    assert response.json() == enum_values

def test_get_type_unknown():
    response = client.get("/model/unknown")
    assert response.status_code == 422  # Unprocessable Entity due to enum validation

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
