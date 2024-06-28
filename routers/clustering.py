import pycaret.clustering as clustering
from pycaret.datasets import get_data

from fastapi import APIRouter

router = APIRouter(
    prefix="/clustering",
    tags=["Clustering"],
    responses={404: {"description": "Not found"}},
)


@router.get("/")
async def clustering_models_list() -> list:
    data = get_data('public_health')
    clustering.setup(data, ignore_features=['Country Name'], session_id=123, log_experiment=True, log_plots=True,
                     experiment_name='health1')
    return clustering.models().index.tolist()
