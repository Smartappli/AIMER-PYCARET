import pytest
from httpx import AsyncClient
from fastapi.testclient import TestClient
from routers import clustering

from fastapi import FastAPI

app = FastAPI()
app.include_router(clustering.router)

client = TestClient(app)


@pytest.mark.anyio
async def test_clustering_models_list():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.get("/clustering/")
    assert response.status_code == 200
    assert isinstance(response.json(), list)
    assert len(response.json()) > 0
