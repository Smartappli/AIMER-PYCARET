import pytest
from httpx import AsyncClient
from fastapi import FastAPI
from routers import time_series

app = FastAPI()
app.include_router(time_series.router)


@pytest.mark.anyio
async def test_time_series_models_list():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.get("/time_series/")
    assert response.status_code == 200
    assert isinstance(response.json(), list)
    assert (
        len(response.json()) > 0
    )  # Ensure there's at least one model in the list


@pytest.mark.asyio
async def test_time_series_model_compare():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.get("/time_series/compare_models")
    assert response.status_code == 200
    result = response.json().get("result")
    assert isinstance(result, list)
    assert (
        len(result) > 0
    )  # Ensure there's at least one model in the comparison
    required_keys = [
        "Model",
        "MASE",
        "RMSSE",
        "MAE",
        "RMSE",
        "MAPE",
        "SMAPE",
        "R2",
        "TT (Sec)",
    ]
    for model_info in result:
        for key in required_keys:
            assert key in model_info  # Ensure all required metrics are present
