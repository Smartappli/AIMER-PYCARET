import os

from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
from enum import Enum
from logging import Logger
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
import pycaret.anomaly as anomaly
import pycaret.classification as classification
import pycaret.clustering as clustering
import pycaret.regression as regression
import pycaret.time_series as time_series
from anyio import to_thread
from fastapi import APIRouter, Depends, FastAPI, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from joblib import Memory
import jwt
from jwt import PyJWTError
from numpy import ndarray
from pandas import DataFrame, Series
from pycaret.loggers.base_logger import BaseLogger
from pydantic import BaseModel
from scipy.sparse import spmatrix
from loguru import logger


# Load environment variables from .env file
load_dotenv()

# Read variables from environment
SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = os.getenv("ALGORITHM")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES"))

# Initialize the logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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


class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    username: Optional[str] = None


class User(BaseModel):
    username: str


class UserInDB(User):
    hashed_password: str


# Initialize the API router
router = APIRouter()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


fake_users_db = {
    "johndoe": {"username": "johndoe", "hashed_password": "fakehashedpassword"}
}


def verify_password(plain_password, hashed_password):
    # Verify password is not empty
    if not plain_password:
        return False
    return plain_password == hashed_password


def get_user(db, username: str):
    if username in db:
        user_dict = db[username]
        return UserInDB(**user_dict)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except PyJWTError:
        raise credentials_exception
    user = get_user(fake_users_db, username=token_data.username)
    if user is None:
        raise credentials_exception
    return user


@router.get("/")
async def root():
    return {"pycaret_version": pycaret.__version__}


@router.post("/token", response_model=Token)
async def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(),
):
    user = get_user(fake_users_db, form_data.username)
    if not user or not verify_password(
        form_data.password, user.hashed_password
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}


@router.get("/secure-endpoint")
async def read_secure_data(current_user: User = Depends(get_current_user)):
    return {"message": f"Hello, {current_user.username}"}


@router.get("/model/{model_type}")
async def get_type(model_type: ModelType):
    pycaret_models = get_pycaret_models()

    if model_type in pycaret_models:
        return pycaret_models[model_type]
    else:
        return {"error": "Model Type Unknown"}


@router.post("/anomaly_detection")
async def anomaly_detection_endpoint(
    setup_params: AnomalyDetectionSetup,
    train_params: AnomalyDetectionParams,
    current_user: TokenData = Depends(get_current_user),
):
    try:
        logger.info("Starting anomaly detection setup and training.")

        # Convert setup_params and train_params to dictionaries
        setup_config = setup_params.dict()
        train_config = train_params.dict()

        # Create an instance of the anomaly detection class
        anomaly_instance = anomaly.AnomalyExperiment()

        # Process data_func if it exists
        if "data_func" in setup_config and callable(setup_config["data_func"]):
            setup_config["data"] = await to_thread.run_sync(
                setup_config.pop("data_func")
            )

        # Perform anomaly detection setup
        setup_result = await to_thread.run_sync(
            anomaly_instance.setup, **setup_config
        )
        logger.info(f"Setup result: {setup_result}")

        # Check if specific models are specified in train_params
        models_to_train = train_config.get("models", None)

        # Perform anomaly detection model training
        if not models_to_train:
            # If no models are specified, use compare_models to evaluate all models
            logger.info(
                "No specific models specified, comparing all available models."
            )

            compare_result = await to_thread.run_sync(
                anomaly_instance.compare_models
            )
            logger.info(f"Training result: {compare_result}")

            result = {
                "setup": setup_result,
                "compare": compare_result,
            }
            logger.info(
                "Anomaly detection setup, and model comparison completed successfully."
            )

        else:
            # Train the specified model(s)
            logger.info(f"Training specified models: {models_to_train}")
            train_result = await to_thread.run_sync(
                anomaly_instance.create_model, **train_config
            )
            logger.info(f"Training result: {train_result}")

            # Perform model evaluation
            evaluate_result = await to_thread.run_sync(
                anomaly_instance.evaluate_model, train_result
            )
            logger.info(f"Evaluation result: {evaluate_result}")

            # Perform model tuning
            tune_result = await to_thread.run_sync(
                anomaly_instance.tune_model, train_result
            )
            logger.info(f"Tuning result: {tune_result}")

            # Perform model plotting
            plot_result = await to_thread.run_sync(
                anomaly_instance.plot_model,
                tune_result,
                plot="confusion_matrix",
            )
            logger.info(f"Plotting result: {plot_result}")

            # Perform model interpretation
            interpret_result = await to_thread.run_sync(
                anomaly_instance.interpret_model, tune_result
            )
            logger.info(f"Interpretation result: {interpret_result}")

            # Finalize the model
            finalize_result = await to_thread.run_sync(
                anomaly_instance.finalize_model, tune_result
            )
            logger.info(f"Finalize result: {finalize_result}")

            # Save the model
            save_path = f"anomaly_model_{current_user.username}.pkl"  # Save the model with the user's username
            save_result = await to_thread.run_sync(
                anomaly_instance.save_model, finalize_result, save_path
            )
            logger.info(f"Model saved at: {save_path}")

            result = {
                "setup": setup_result,
                "train": train_result,
                "evaluate": evaluate_result,
                "tune": tune_result,
                "plot": plot_result,
                "interpret": interpret_result,
                "finalize": finalize_result,
                "save": save_result,
            }
            logger.info(
                "Anomaly detection setup, training, evaluation, tuning, plotting, interpretation, finalization, and saving completed successfully."
            )

        return result

    except Exception as e:
        logger.error(f"An error occurred during anomaly detection: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/classification")
async def classification_endpoint(
    setup_params: ClassificationSetup,
    train_params: ClassificationParams,
    current_user: TokenData = Depends(get_current_user),
):
    try:
        logger.info("Starting classification setup and training.")

        # Convert setup_params and train_params to dictionaries
        setup_config = setup_params.dict()
        train_config = train_params.dict()

        # Create an instance of the classification class
        classification_instance = classification.ClassificationExperiment()

        # Process data_func if it exists
        if "data_func" in setup_config and callable(setup_config["data_func"]):
            setup_config["data"] = await to_thread.run_sync(
                setup_config.pop("data_func")
            )

        # Perform classification setup
        setup_result = await to_thread.run_sync(
            classification_instance.setup, **setup_config
        )
        logger.info(f"Setup result: {setup_result}")

        # Perform classification model training
        train_result = await to_thread.run_sync(
            classification_instance.create_model, **train_config
        )
        logger.info(f"Training result: {train_result}")

        result = {"setup": setup_result, "train": train_result}
        logger.info("Classification setup and training completed successfully.")
        return result

    except Exception as e:
        logger.error(f"An error occurred during classification: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/clustering")
async def clustering_endpoint(
    setup_params: ClusteringSetup,
    train_params: ClusteringParams,
    current_user: TokenData = Depends(get_current_user),
):
    try:
        logger.info("Starting clustering setup and training.")

        # Convert setup_params and train_params to dictionaries
        setup_config = setup_params.dict()
        train_config = train_params.dict()

        # Create an instance of the clustering class
        clustering_instance = clustering.ClusteringExperiment()

        # Process data_func if it exists
        if "data_func" in setup_config and callable(setup_config["data_func"]):
            setup_config["data"] = await to_thread.run_sync(
                setup_config.pop("data_func")
            )

        # Perform clustering setup
        setup_result = await to_thread.run_sync(
            clustering_instance.setup, **setup_config
        )
        logger.info(f"Setup result: {setup_result}")

        # Perform clustering model training
        train_result = await to_thread.run_sync(
            clustering_instance.create_model, **train_config
        )
        logger.info(f"Training result: {train_result}")

        result = {"setup": setup_result, "train": train_result}
        logger.info("Clustering setup and training completed successfully.")
        return result

    except Exception as e:
        logger.error(f"An error occurred during clustering: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/regression")
async def regression_endpoint(
    setup_params: RegressionSetup,
    train_params: RegressionParams,
    current_user: TokenData = Depends(get_current_user),
):
    try:
        logger.info("Starting regression setup and training.")
        setup_config = setup_params.dict()
        train_config = train_params.dict()

        # Create an instance of the clustering class
        regression_instance = regression.RegressiongExperiment()

        # Process data_func if it exists
        if "data_func" in setup_config and callable(setup_config["data_func"]):
            setup_config["data"] = await to_thread.run_sync(
                setup_config.pop("data_func")
            )

        # Perform regression setup
        setup_result = await to_thread.run_sync(
            regression_instance.setup, **setup_config
        )
        logger.info(f"Setup result: {setup_result}")

        # Perform clustering model training
        train_result = await to_thread.run_sync(
            regression_instance.create_model, **train_config
        )
        logger.info(f"Training result: {train_result}")

        result = {"setup": setup_result, "train": train_result}
        logger.info("Regression setup and training completed successfully.")
        return result

    except Exception as e:
        logger.error(f"An error occurred during regression: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/time_series")
async def time_series_endpoint(
    setup_params: TimeSeriesSetup,
    train_params: TimeSeriesParams,
    current_user: TokenData = Depends(get_current_user),
):
    try:
        logger.info("Starting time series setup and training.")
        setup_config = setup_params.dict()
        train_config = train_params.dict()

        # Create an instance of the time_series class
        time_series_instance = time_series.TSForecastingExperiment()

        # Process data_func if it exists
        if "data_func" in setup_config and callable(setup_config["data_func"]):
            setup_config["data"] = await to_thread.run_sync(
                setup_config.pop("data_func")
            )

        # Perform time series setup
        setup_result = await to_thread.run_sync(
            time_series_instance.setup, **setup_config
        )
        logger.info(f"Setup result: {setup_result}")

        # Perform time series model training
        train_result = await to_thread.run_sync(
            time_series_instance.create_model, **train_config
        )
        logger.info(f"Training result: {train_result}")

        result = {"setup": setup_result, "train": train_result}
        logger.info("Time series setup and training completed successfully.")
        return result

    except Exception as e:
        logger.error(f"An error occurred during time series: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Create the FastAPI app and include the router
app = FastAPI()
app.include_router(router)
