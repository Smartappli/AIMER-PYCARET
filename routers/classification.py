from typing import Any, Dict, List

import pycaret.classification as classification
from pycaret.datasets import get_data
from pandas import DataFrame
from pydantic import BaseModel
from fastapi import APIRouter


class Result(BaseModel):
    result: DataFrame

    class Config:
        arbitrary_types_allowed = True


router = APIRouter(
    prefix="/classification",
    tags=["Classification"],
    responses={404: {"description": "Not found"}},
)


@router.get("/")
async def classification_models_list() -> List:
    data = get_data("juice")
    exp = classification.ClassificationExperiment()
    exp.setup(
        data,
        target="Purchase",
        session_id=123,
        log_experiment=True,
        experiment_name="juice1",
    )
    return exp.models().index.tolist()


@router.get("/compare_models")
async def model_compare() -> Dict:
    data = get_data("juice")
    exp = classification.ClassificationExperiment()
    exp.setup(
        data,
        target="Purchase",
        session_id=123,
        log_experiment=True,
        experiment_name="juice1",
    )

    # Compare all models
    models = exp.compare_models()

    # Retrieve the latest displayed table with model comparison metrics
    comparison_results = exp.pull()

    # Define the metrics you want to extract
    metrics = [
        "Model",
        "Accuracy",
        "AUC",
        "Recall",
        "Prec.",
        "F1",
        "Kappa",
        "MCC",
        "TT (Sec)",
    ]

    # Add an 'ID' column to the DataFrame
    comparison_results.reset_index(inplace=True)
    comparison_results.rename(columns={"index": "ID"}, inplace=True)

    # Check which metrics are available in the results
    available_metrics = ["ID"] + [
        metric for metric in metrics if metric in comparison_results.columns
    ]

    # Extracting detailed information from the comparison results
    model_info = comparison_results[available_metrics].to_dict(orient="records")

    return {"result": model_info}


@router.post("/create_model")
async def create_model() -> Any:
    data = get_data("juice")
    exp = classification.ClassificationExperiment()
    exp.setup(
        data,
        target="Purchase",
        session_id=123,
        log_experiment=True,
        experiment_name="juice1",
    )

    exp.create_model("lr", return_train_score=True)
