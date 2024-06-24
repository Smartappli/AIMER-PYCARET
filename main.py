from enum import Enum
from logging import Logger
from typing import Any, Callable, Dict, List, Optional

import pycaret.anomaly as anomaly
import pycaret.classification as classification
import pycaret.clustering as clustering
import pycaret.regression as regression
import pycaret.time_series as time_series
from anyio import to_thread
from fastapi import FastAPI
from joblib import Memory
from numpy import ndarray
from pandas import DataFrame, Series
from pycaret.loggers.base_logger import BaseLogger
from pydantic import BaseModel
from scipy.sparse import spmatrix
from loguru import logger

app = FastAPI()


class ModelType(str, Enum):
    """Enumeration for different types of models."""
    
    anomaly_detection = "anomaly detection"
    classification = "classification"
    clustering = "clustering"
    regression = "regression"
    time_series = "time series"


def get_pycaret_models() -> Dict[ModelType, list]:
    """
    Retrieve available estimators for different model types in PyCaret.

    This function queries the available estimators for various model types 
    such as anomaly detection, classification, clustering, regression, 
    and time series from the PyCaret library.

    Returns:
        Dict[ModelType, list]: A dictionary where the keys are model types 
        (anomaly detection, classification, clustering, regression, 
        and time series) and the values are lists of available estimators 
        for each model type.
    """
    models = {
        ModelType.anomaly_detection: anomaly.models(),
        ModelType.classification: classification.models(),
        ModelType.clustering: pycaret.clustering.models(),
        ModelType.regression: pycaret.regression.models(),
        ModelType.time_series: pycaret.time_series.models(),
    }
    return models


class AnomalyDetectionSetup(BaseModel):
    """Setup parameters for anomaly detection models."""

    data: Optional[dict | list | tuple | ndarray | spmatrix | DataFrame] = None
    data_func: Optional[
        Callable[[], dict | list | tuple | ndarray | spmatrix | DataFrame]
    ] = None
    index: bool | int | str | list | tuple | ndarray | Series = True
    ordinal_features: Optional[Dict[str, list]] = None
    numeric_features: Optional[List[str]] = None
    categorical_features: Optional[List[str]] = None
    date_features: Optional[List[str]] = None
    text_features: Optional[List[str]] = None
    ignore_features: Optional[List[str]] = None
    keep_features: Optional[List[str]] = None
    preprocess: bool = True
    create_date_columns: List[str] = ["day", "month", "year"]
    imputation_type: Optional[str] = "simple"
    numeric_imputation: str = "mean"
    categorical_imputation: str = "mode"
    text_features_method: str = "tf-idf"
    max_encoding_ohe: int = -1
    encoding_method: Optional[Any] = None
    rare_to_value: Optional[float] = None
    rare_value: str = "rare"
    polynomial_features: bool = False
    polynomial_degree: int = 2
    low_variance_threshold: Optional[float] = None
    group_features: Optional[dict] = None
    drop_groups: bool = False
    remove_multicollinearity: bool = False
    multicollinearity_threshold: float = 0.9
    bin_numeric_features: Optional[List[str]] = None
    remove_outliers: bool = False
    outliers_method: str = "iforest"
    outliers_threshold: float = 0.05
    transformation: bool = False
    transformation_method: str = "yeo-johnson"
    normalize: bool = False
    normalize_method: str = "zscore"
    pca: bool = False
    pca_method: str = "linear"
    pca_components: Optional[int | float | str] = None
    custom_pipeline: Optional[Any] = None
    custom_pipeline_position: int = -1
    n_jobs: Optional[int] = -1
    use_gpu: bool = False
    html: bool = True
    session_id: Optional[int] = None
    system_log: bool | str | Logger = True
    log_experiment: bool | str | BaseLogger | List[str | BaseLogger] = False
    experiment_name: Optional[str] = None
    experiment_custom_tags: Optional[Dict[str, Any]] = None
    log_plots: bool | list = False
    log_profile: bool = False
    log_data: bool = False
    verbose: bool = True
    memory: bool | str | Memory = True
    profile: bool = False
    profile_kwargs: Optional[Dict[str, Any]] = None


class AnomalyDetectionParams(BaseModel):
    """Parameters for training anomaly detection models."""

    model: str | Any
    fraction: float = 0.05
    verbose: bool = True
    fit_kwargs: Optional[dict] = None
    experiment_custom_tags: Optional[Dict[str, Any]] = None


class ClassificationSetup(BaseModel):
    """Setup parameters for classification models."""

    data: Optional[dict | list | tuple | ndarray | spmatrix | DataFrame] = None
    data_func: Optional[
        Callable[[], dict | list | tuple | ndarray | spmatrix | DataFrame]
    ] = None
    target: int | str | list | tuple | ndarray | Series = -1
    index: bool | int | str | list | tuple | ndarray | Series = True
    train_size: float = 0.7
    test_data: Optional[
        dict | list | tuple | ndarray | spmatrix | DataFrame
    ] = None
    ordinal_features: Optional[Dict[str, list]] = None
    numeric_features: Optional[List[str]] = None
    categorical_features: Optional[List[str]] = None
    date_features: Optional[List[str]] = None
    text_features: Optional[List[str]] = None
    ignore_features: Optional[List[str]] = None
    keep_features: Optional[List[str]] = None
    preprocess: bool = True
    create_date_columns: List[str] = ["day", "month", "year"]
    imputation_type: Optional[str] = "simple"
    numeric_imputation: str = "mean"
    categorical_imputation: str = "mode"
    iterative_imputation_iters: int = 5
    numeric_iterative_imputer: str | Any = "lightgbm"
    categorical_iterative_imputer: str | Any = "lightgbm"
    text_features_method: str = "tf-idf"
    max_encoding_ohe: int = 25
    encoding_method: Optional[Any] = None
    rare_to_value: Optional[float] = None
    rare_value: str = "rare"
    polynomial_features: bool = False
    polynomial_degree: int = 2
    low_variance_threshold: Optional[float] = None
    group_features: Optional[dict] = None
    drop_groups: bool = False
    remove_multicollinearity: bool = False
    multicollinearity_threshold: float = 0.9
    bin_numeric_features: Optional[List[str]] = None
    remove_outliers: bool = False
    outliers_method: str = "iforest"
    outliers_threshold: float = 0.05
    fix_imbalance: bool = False
    fix_imbalance_method: str | Any = "SMOTE"
    transformation: bool = False
    transformation_method: str = "yeo-johnson"
    normalize: bool = False
    normalize_method: str = "zscore"
    pca: bool = False
    pca_method: str = "linear"
    pca_components: Optional[int | float | str] = None
    feature_selection: bool = False
    feature_selection_method: str = "classic"
    feature_selection_estimator: str | Any = "lightgbm"
    n_features_to_select: int | float = 0.2
    custom_pipeline: Optional[Any] = None
    custom_pipeline_position: int = -1
    data_split_shuffle: bool = True
    data_split_stratify: bool | List[str] = True
    fold_strategy: str | Any = "stratifiedkfold"
    fold: int = 10
    fold_shuffle: bool = False
    fold_groups: Optional[str | DataFrame] = None
    n_jobs: Optional[int] = -1
    use_gpu: bool = False
    html: bool = True
    session_id: Optional[int] = None
    system_log: bool | str | Logger = True
    log_experiment: bool | str | BaseLogger | List[str | BaseLogger] = False
    experiment_name: Optional[str] = None
    experiment_custom_tags: Optional[Dict[str, Any]] = None
    log_plots: bool | list = False
    log_profile: bool = False
    log_data: bool = False
    engine: Optional[Any] = None
    verbose: bool = True
    memory: bool | str | Memory = True
    profile: bool = False
    profile_kwargs: Optional[Dict[str, Any]] = None


class ClassificationParams(BaseModel):
    """Parameters for training classification models."""

    model: str | Any
    verbose: bool = True
    fit_kwargs: Optional[dict] = None
    groups: Optional[DataFrame | ndarray | Series] = None
    experiment_custom_tags: Optional[Dict[str, Any]] = None
    probability_threshold: Optional[float] = None
    engine: Optional[str] = None
    verbose: bool = True
    return_train_score: bool = False


class ClusteringSetup(BaseModel):
    data: Optional[dict | list | tuple | ndarray | spmatrix | DataFrame] = None
    data_func: Optional[
        Callable[[], dict | list | tuple | ndarray | spmatrix | DataFrame]
    ] = None
    index: bool | int | str | list | tuple | ndarray | Series = True
    ordinal_features: Optional[Dict[str, list]] = None
    numeric_features: Optional[List[str]] = None
    categorical_features: Optional[List[str]] = None
    date_features: Optional[List[str]] = None
    text_features: Optional[List[str]] = None
    ignore_features: Optional[List[str]] = None
    keep_features: Optional[List[str]] = None
    preprocess: bool = True
    create_date_columns: List[str] = ["day", "month", "year"]
    imputation_type: Optional[str] = "simple"
    numeric_imputation: str = "mean"
    categorical_imputation: str = "mode"
    text_features_method: str = "tf-idf"
    max_encoding_ohe: int = -1
    encoding_method: Optional[Any] = None
    rare_to_value: Optional[float] = None
    rare_value: str = "rare"
    polynomial_features: bool = False
    polynomial_degree: int = 2
    low_variance_threshold: Optional[float] = None
    remove_multicollinearity: bool = False
    multicollinearity_threshold: float = 0.9
    bin_numeric_features: Optional[List[str]] = None
    remove_outliers: bool = False
    outliers_method: str = "iforest"
    outliers_threshold: float = 0.05
    transformation: bool = False
    transformation_method: str = "yeo-johnson"
    normalize: bool = False
    normalize_method: str = "zscore"
    pca: bool = False
    pca_method: str = "linear"
    pca_components: Optional[int | float | str] = None
    custom_pipeline: Optional[Any] = None
    custom_pipeline_position: int = -1
    n_jobs: Optional[int] = -1
    use_gpu: bool = False
    html: bool = True
    session_id: Optional[int] = None
    system_log: bool | str | Logger = True
    log_experiment: bool | str | BaseLogger | List[str | BaseLogger] = False
    experiment_name: Optional[str] = None
    experiment_custom_tags: Optional[Dict[str, Any]] = None
    log_plots: bool | list = False
    log_profile: bool = False
    log_data: bool = False
    verbose: bool = True
    memory: bool | str | Memory = True
    profile: bool = False
    profile_kwargs: Optional[Dict[str, Any]] = None


class ClusteringParams(BaseModel):
    model: str | Any
    num_clusters: int = 4
    ground_truth: Optional[str] = None
    round: int = 4
    fit_kwargs: Optional[dict] = None
    verbose: bool = True
    experiment_custom_tags: Optional[Dict[str, Any]] = None
    engine: Optional[str] = None


class RegressionSetup(BaseModel):
    data: Optional[dict | list | tuple | ndarray | spmatrix | DataFrame] = None
    data_func: Optional[
        Callable[[], dict | list | tuple | ndarray | spmatrix | DataFrame]
    ] = None
    target: int | str | list | tuple | ndarray | Series = -1
    index: bool | int | str | list | tuple | ndarray | Series = True
    train_size: float = 0.7
    test_data: Optional[
        dict | list | tuple | ndarray | spmatrix | DataFrame
    ] = None
    ordinal_features: Optional[Dict[str, list]] = None
    numeric_features: Optional[List[str]] = None
    categorical_features: Optional[List[str]] = None
    date_features: Optional[List[str]] = None
    text_features: Optional[List[str]] = None
    ignore_features: Optional[List[str]] = None
    keep_features: Optional[List[str]] = None
    preprocess: bool = True
    create_date_columns: List[str] = ["day", "month", "year"]
    imputation_type: Optional[str] = "simple"
    numeric_imputation: str = "mean"
    categorical_imputation: str = "mode"
    iterative_imputation_iters: int = 5
    numeric_iterative_imputer: str | Any = "lightgbm"
    categorical_iterative_imputer: str | Any = "lightgbm"
    text_features_method: str = "tf-idf"
    max_encoding_ohe: int = 25
    encoding_method: Optional[Any] = None
    rare_to_value: Optional[float] = None
    rare_value: str = "rare"
    polynomial_features: bool = False
    polynomial_degree: int = 2
    low_variance_threshold: Optional[float] = None
    group_features: Optional[dict] = None
    drop_groups: bool = False
    remove_multicollinearity: bool = False
    multicollinearity_threshold: float = 0.9
    bin_numeric_features: Optional[List[str]] = None
    remove_outliers: bool = False
    outliers_method: str = "iforest"
    outliers_threshold: float = 0.05
    transformation: bool = False
    transformation_method: str = "yeo-johnson"
    normalize: bool = False
    normalize_method: str = "zscore"
    pca: bool = False
    pca_method: str = "linear"
    pca_components: Optional[int | float | str] = None
    feature_selection: bool = False
    feature_selection_method: str = "classic"
    feature_selection_estimator: str | Any = "lightgbm"
    n_features_to_select: int | float = 0.2
    transform_target: bool = False
    transform_target_method: str = "yeo-johnson"
    custom_pipeline: Optional[Any] = None
    custom_pipeline_position: int = -1
    data_split_shuffle: bool = True
    data_split_stratify: bool | List[str] = False
    fold_strategy: str | Any = "kfold"
    fold: int = 10
    fold_shuffle: bool = False
    fold_groups: Optional[str | DataFrame] = None
    n_jobs: Optional[int] = -1
    use_gpu: bool = False
    html: bool = True
    session_id: Optional[int] = None
    system_log: bool | str | Logger = True
    log_experiment: bool | str | BaseLogger | List[str | BaseLogger] = False
    experiment_name: Optional[str] = None
    experiment_custom_tags: Optional[Dict[str, Any]] = None
    log_plots: bool | list = False
    log_profile: bool = False
    log_data: bool = False
    engine: Optional[Dict[str, str]] = None
    verbose: bool = True
    memory: bool | str | Memory = True
    profile: bool = False
    profile_kwargs: Optional[Dict[str, Any]] = None


class RegressionParams(BaseModel):
    estimator: str | Any
    fold: Optional[int | Any] = None
    round: int = 4
    cross_validation: bool = True
    fit_kwargs: Optional[dict] = None
    groups: Optional[str | Any] = None
    experiment_custom_tags: Optional[Dict[str, Any]] = None
    engine: Optional[str] = None
    verbose: bool = True
    return_train_score: bool = False


class TimeSeriesSetup(BaseModel):
    data: Optional[Series | DataFrame] = None
    data_func: Optional[Callable[[], Series | DataFrame]] = None
    target: Optional[str] = None
    index: Optional[str] = None
    ignore_features: Optional[List] = None
    numeric_imputation_target: Optional[str | int | float] = None
    numeric_imputation_exogenous: Optional[str | int | float] = None
    transform_target: Optional[str] = None
    transform_exogenous: Optional[str] = None
    scale_target: Optional[str] = None
    scale_exogenous: Optional[str] = None
    fe_target_rr: Optional[list] = None
    fe_exogenous: Optional[list] = None
    fold_strategy: str | Any = "expanding"
    fold: int = 3
    fh: Optional[List[int] | int | ndarray | ForecastingHorizon] = 1
    hyperparameter_split: str = "all"
    seasonal_period: Optional[List[int | str] | int | str] = None
    ignore_seasonality_test: bool = False
    sp_detection: str = "auto"
    max_sp_to_consider: Optional[int] = 60
    remove_harmonics: bool = False
    harmonic_order_method: str = "harmonic_max"
    num_sps_to_use: int = 1
    seasonality_type: str = "mul"
    point_alpha: Optional[float] = None
    coverage: float | List[float] = 0.9
    enforce_exogenous: bool = True
    n_jobs: Optional[int] = -1
    use_gpu: bool = False
    custom_pipeline: Optional[Any] = None
    html: bool = True
    session_id: Optional[int] = None
    system_log: bool | str | Logger = True
    log_experiment: bool | str | BaseLogger | List[str | BaseLogger] = False
    experiment_name: Optional[str] = None
    experiment_custom_tags: Optional[Dict[str, Any]] = None
    log_plots: bool | list = False
    log_profile: bool = False
    log_data: bool = False
    engine: Optional[Dict[str, str]] = None
    verbose: bool = True
    profile: bool = False
    profile_kwargs: Optional[Dict[str, Any]] = None
    fig_kwargs: Optional[Dict[str, Any]] = None


class TimeSeriesParams(BaseModel):
    estimator: str | Any
    fold: Optional[int | Any] = None
    round: int = 4
    cross_validation: bool = True
    fit_kwargs: Optional[dict] = None
    engine: Optional[str] = None
    verbose: bool = True


@app.get("/")
async def root():
    return {"pycaret_version": pycaret.__version__}


@app.get("/model/{model_type}")
async def get_type(model_type: ModelType):
    pycaret_models = get_pycaret_models()

    if model_type in pycaret_models:
        return pycaret_models[model_type]
    else:
        return {"error": "Model Type Unknown"}


@app.post("/anomaly_detection")
async def anomaly_detection_endpoint(
    setup_params: AnomalyDetectionSetup, train_params: AnomalyDetectionParams
):
    try:
        logger.info("Starting anomaly detection setup and training.")
        setup_config = setup_params.dict()
        train_config = train_params.dict()

        if callable(setup_config.get("data_func")):
            setup_config["data"] = setup_config.pop("data_func")()

        setup_config.pop("data_func", None)

        setup_future = to_thread.run_sync(anomaly.setup, **setup_config)
        setup_result = await setup_future

        train_future = to_thread.run_sync(anomaly.create_model, **train_config)
        train_result = await train_future

        result = {"setup": setup_result, "train": train_result}
        logger.info("Anomaly detection setup and training completed successfully.")
        return result
    except Exception as e:
        logger.error(f"An error occurred during anomaly detection: {str(e)}")
        return {"error": str(e)}


@app.post("/classification")
async def classification_endpoint(
    setup_params: ClassificationSetup, train_params: ClassificationParams
):
    try:
        logger.info("Starting classification setup and training.")
        setup_config = setup_params.dict()
        train_config = train_params.dict()

        if callable(setup_config.get("data_func")):
            setup_config["data"] = setup_config.pop("data_func")()

        setup_config.pop("data_func", None)

        setup_future = to_thread.run_sync(classification.setup, **setup_config)
        setup_result = await setup_future

        train_future = to_thread.run_sync(classification.create_model, **train_config)
        train_result = await train_future

        result = {"setup": setup_result, "train": train_result}
        logger.info("Classification setup and training completed successfully.")
        return result
    except Exception as e:
        logger.error(f"An error occurred during classification: {str(e)}")
        return {"error": str(e)}


@app.post("/clustering")
async def clustering_endpoint(
    setup_params: ClusteringSetup, train_params: ClusteringParams
):
    try:
        logger.info("Starting clustering setup and training.")
        setup_config = setup_params.dict()
        train_config = train_params.dict()

        if callable(setup_config.get("data_func")):
            setup_config["data"] = setup_config.pop("data_func")()

        setup_config.pop("data_func", None)

        setup_future = to_thread.run_sync(clustering.setup, **setup_config)
        setup_result = await setup_future

        train_future = to_thread.run_sync(clustering.create_model, **train_config)
        train_result = await train_future

        result = {"setup": setup_result, "train": train_result}
        logger.info("Clustering setup and training completed successfully.")
        return result
    except Exception as e:
        logger.error(f"An error occurred during clustering: {str(e)}")
        return {"error": str(e)}


@app.post("/regression")
async def regression_endpoint(
    setup_params: RegressionSetup, train_params: RegressionParams
):
    try:
        logger.info("Starting regression setup and training.")
        setup_config = setup_params.dict()
        train_config = train_params.dict()

        if callable(setup_config.get("data_func")):
            setup_config["data"] = setup_config.pop("data_func")()

        setup_config.pop("data_func", None)

        setup_future = to_thread.run_sync(regression.setup, **setup_config)
        setup_result = await setup_future

        train_future = to_thread.run_sync(regression.create_model, **train_config)
        train_result = await train_future

        result = {"setup": setup_result, "train": train_result}
        logger.info("Regression setup and training completed successfully.")
        return result
    except Exception as e:
        logger.error(f"An error occurred during regression: {str(e)}")
        return {"error": str(e)}


@app.post("/time_series")
async def classification_endpoint(
    setup_params: TimeSeriesSetup, train_params: TimeSeriesParams
):
    try:
        logger.info("Starting time series setup and training.")
        setup_config = setup_params.dict()
        train_config = train_params.dict()

        if callable(setup_config.get("data_func")):
            setup_config["data"] = setup_config.pop("data_func")()

        setup_config.pop("data_func", None)

        setup_future = to_thread.run_sync(time_series.setup, **setup_config)
        setup_result = await setup_future

        train_future = to_thread.run_sync(tiume_series.create_model, **train_config)
        train_result = await train_future

        result = {"setup": setup_result, "train": train_result}
            logger.info("Time series setup and training completed successfully.")
        return result
    except Exception as e:
        logger.error(f"An error occurred during time series: {str(e)}")
        return {"error": str(e)}
