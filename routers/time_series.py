import pycaret.time_series as time_series
from pycaret.datasets import get_data

from fastapi import APIRouter

router = APIRouter(
    prefix="/time_series",
    tags=["Time series"],
    responses={404: {"description": "Not found"}},
)


@router.get("/")
async def time_series_models_list() -> list:
    data = get_data('airline')
    exp = time_series.TSForecastingExperiment
    exp.setup(data, fh=3, session_id=123)
    return exp.models().index.tolist()


@router.get("/compare_models")
async def model_compare() -> dict:
    data = get_data('airline')
    exp = time_series.TSForecastingExperiment()
    exp.setup(data, fh=3, session_id=123)

    # Compare all models
    models = exp.compare_models()

    # Retrieve the latest displayed table with model comparison metrics
    comparison_results = exp.pull()

    # Define the metrics you want to extract
    metrics = ['Model', 'MASE', 'RMSSE', 'MAE', 'RMSE', 'MAPE', 'SMAPE', 'R2', 'TT (Sec)']

    # Add an 'ID' column to the DataFrame
    comparison_results.reset_index(inplace=True)
    comparison_results.rename(columns={'index': 'ID'}, inplace=True)

    # Check which metrics are available in the results
    available_metrics = ['ID'] + [metric for metric in metrics if metric in comparison_results.columns]

    # Extracting detailed information from the comparison results
    model_info = comparison_results[available_metrics].to_dict(orient='records')

    return {"result": model_info}