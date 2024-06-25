# test_authentication.py
import pytest
from fastapi import status


def test_login_success(test_app, test_user):
    response = test_app.post("/token", data=test_user)
    assert response.status_code == status.HTTP_200_OK
    assert "access_token" in response.json()
    assert response.json()["token_type"] == "bearer"


def test_login_wrong_password(test_app, test_user):
    test_user["password"] = "wrongpassword"
    response = test_app.post("/token", data=test_user)
    assert response.status_code == status.HTTP_401_UNAUTHORIZED
    assert response.json()["detail"] == "Incorrect username or password"
