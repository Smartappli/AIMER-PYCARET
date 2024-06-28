import pytest
from httpx import AsyncClient
from fastapi.testclient import TestClient
from routers import anomaly_detection

from fastapi import FastAPI

app = FastAPI()
app.include_router(anomaly_detection.router)

client = TestClient(app)


@pytest.mark.anyio
async def test_anomaly_detection_models_list():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.get("/anomaly_detection/")
    assert response.status_code == 200
    assert isinstance(response.json(), list)
    assert len(response.json()) > 0
