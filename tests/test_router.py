# test_router.py
import pytest
from fastapi import status
from app.router import ModelType

def test_root(test_app):
    response = test_app.get("/")
    assert response.status_code == status.HTTP_200_OK
    assert "pycaret_version" in response.json()

def test_get_type(test_app):
    response = test_app.get("/model/anomaly_detection")
    assert response.status_code == status.HTTP_200_OK
    assert isinstance(response.json(), list)
    assert ModelType.anomaly_detection.value in response.json()

def test_anomaly_detection_endpoint(test_app, test_user):
    setup_payload = {"data": {"key": "value"}}  # Adjust payload as per your endpoint
    response = test_app.post("/anomaly_detection", json=setup_payload, headers={"Authorization": f"Bearer {create_access_token({'sub': test_user['username']})}"})
    assert response.status_code == status.HTTP_200_OK
    assert "setup" in response.json()
    assert "train" in response.json()  # Adjust assertions based on expected output

def test_classification_endpoint(test_app, test_user):
    setup_payload = {"data": {"key": "value"}}  # Adjust payload as per your endpoint
    response = test_app.post("/classification", json=setup_payload, headers={"Authorization": f"Bearer {create_access_token({'sub': test_user['username']})}"})
    assert response.status_code == status.HTTP_200_OK
    assert "setup" in response.json()
    assert "train" in response.json()  # Adjust assertions based on expected output
