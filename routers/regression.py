import pycaret.regression as regression
from pycaret.datasets import get_data

from fastapi import APIRouter

router = APIRouter(
    prefix="/regression",
    tags=["regression"],
    responses={404: {"description": "Not found"}},
)


@router.get("/")
async def regression_models_list() -> list:
    data = get_data('insurance')
    regression.setup(data, target='charges', session_id=123, log_experiment=True, experiment_name='insurance1')
    return regression.models().index.tolist()


@router.get("/compare_models")
async def model_compare() -> dict:
    data = get_data('insurance')
    regression.setup(data, target='charges', session_id=123, log_experiment=True, experiment_name='insurance1')

    # Compare all models
    models = regression.compare_models()

    # Retrieve the latest displayed table with model comparison metrics
    comparison_results = regression.pull()

    # Define the metrics you want to extract
    metrics = ['Model', 'MAE', 'MSE', 'RMSE', 'R2', 'RMSLE', 'MAPE', 'TT (Sec)']

    # Add an 'ID' column to the DataFrame
    comparison_results.reset_index(inplace=True)
    comparison_results.rename(columns={'index': 'ID'}, inplace=True)

    # Check which metrics are available in the results
    available_metrics = ['ID'] + [metric for metric in metrics if metric in comparison_results.columns]

    # Extracting detailed information from the comparison results
    model_info = comparison_results[available_metrics].to_dict(orient='records')

    return {"result": model_info}

