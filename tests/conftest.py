import pytest
from fastapi.testclient import TestClient

from main import app


@pytest.fixture(scope="module")
def test_app():
    client = TestClient(app)
    yield client


@pytest.fixture(scope="module")
def test_user():
    return {"username": "johndoe", "password": "testpassword"}
