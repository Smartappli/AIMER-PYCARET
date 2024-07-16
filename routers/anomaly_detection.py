from typing import Any, List

import pycaret.anomaly as anomaly
from pycaret.datasets import get_data

from fastapi import APIRouter

router = APIRouter(
    prefix="/anomaly_detection",
    tags=["Anomaly detection"],
    responses={404: {"description": "Not found"}},
)


@router.get("/")
async def anomaly_detection_models_list() -> List:
    data = get_data("anomaly")
    exp = anomaly.AnomalyExperiment()
    exp.setup(data, session_id=123)
    return exp.models().index.tolist()


@router.post("/create_model")
async def create_model() -> Any:
    data = get_data("juice")
    anomaly.setup(data, session_id=123)

    anomaly.create_model("lr", return_train_score=True)
