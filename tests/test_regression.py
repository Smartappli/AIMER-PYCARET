import pytest
from httpx import AsyncClient
from fastapi import FastAPI
from routers import regression

app = FastAPI()
app.include_router(regression.router)

@pytest.mark.anyio
async def test_regression_models_list():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.get("/regression/")
    assert response.status_code == 200
    assert isinstance(response.json(), list)
    assert len(response.json()) > 0

@pytest.mark.anyio
async def test_regression_model_compare():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.get("/regression/compare_models")
    assert response.status_code == 200
    assert "result" in response.json()
    assert isinstance(response.json()["result"], list)
    assert len(response.json()["result"]) > 0
    expected_keys = ['ID', 'Model', 'MAE', 'MSE', 'RMSE', 'R2', 'RMSLE', 'MAPE', 'TT (Sec)']
    for model_info in response.json()["result"]:
        for key in expected_keys:
            assert key in model_info