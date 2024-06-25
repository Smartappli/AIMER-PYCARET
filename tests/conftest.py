import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.router import fake_users_db, verify_password, create_access_token

@pytest.fixture(scope="module")
def test_app():
    client = TestClient(app)
    yield client

@pytest.fixture(scope="module")
def test_user():
    return {"username": "johndoe", "password": "testpassword"}