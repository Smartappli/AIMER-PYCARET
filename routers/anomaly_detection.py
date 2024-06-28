import pycaret.anomaly as anomaly
from pycaret.datasets import get_data

from fastapi import APIRouter

router = APIRouter(
    prefix="/anomaly_detection",
    tags=["Anomaly detection"],
    responses={404: {"description": "Not found"}},
)


@router.get("/")
async def anomaly_detection_models_list() -> list:
    data = get_data('anomaly')
    anomaly.setup(data, session_id = 123)
    return anomaly.models().index.tolist()
