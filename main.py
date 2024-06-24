from enum import Enum
from logging import Logger
from typing import Any, Callable, Dict, List, Optional, Union

import pycaret
from anyio import to_thread
from fastapi import FastAPI
from joblib import Memory
from numpy import ndarray
from pandas import DataFrame, Series
from pycaret.loggers.base_logger import BaseLogger
from pydantic import BaseModel
from scipy.sparse import spmatrix

app = FastAPI()

class ModelType(str, Enum):
    """Enumeration for different types of models."""
    anomaly_detection = "anomaly detection"
    classification = "classification"
    clustering = "clustering"
    regression = "regression"
    time_series = "time series"

class ModelAnomalyDetection(str, Enum):
    """Enumeration for different anomaly detection models."""
    abod = "Angle-base Outlier Detection"
    cluster = "Clustering-Based Local Outlier"
    cof = "Connectivity-Based Outlier Factor"
    histogram = "Histogram-based Outlier Detection"
    iforest = "Isolation Forest"
    knn = "k-Nearest Neighbors Detector"
    lof = "Local Outlier Factor"
    svm = "One-class SVM detector"
    pca = "Principal Component Analysis"
    mcd = "Minimum Covariance Determinant"
    sod = "Subspace Outlier Detection"
    sos = "Stochastic Outlier Selection"

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

class ModelClassification(str, Enum):
    """Enumeration for different classification models."""
    lr = "Logistic Regression"
    knn = "K Neighbors Classifier"
    nb = "Naive Bayes"
    dt = "Decision Tree Classifier"
    svm = "SVM - Linear Kernel"
    rbfsvm = "SVM - Radial Kernel"
    gpc = "Gaussian Process Classifier"
    mlp = "MLP Classifier"
    ridge = "Ridge Classifier"
    rf = "Random Forest Classifier"
    qda = "Quadratic Discriminant Analysis"
    ada = "Ada Boost Classifier"
    gbc = "Gradient Boosting Classifier"
    lda = "Linear Discriminant Analysis"
    et = "Extra Trees Classifier"
    xgboost = "Extreme Gradient Boosting"
    lightgbm = "Light Gradient Boosting Machine"
    catboost = "CatBoost Classifier"

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

@app.post("/setup/anomaly_detection/")
async def setup_anomaly_detection(setup_params: AnomalyDetectionSetup):
    """Setup anomaly detection models with the provided parameters."""
    result = await to_thread.run_sync(pycaret.anomaly.setup, **setup_params.dict())
    return result

@app.post("/train/anomaly_detection/")
async def train_anomaly_detection(params: AnomalyDetectionParams):
    """Train anomaly detection models with the provided parameters."""
    result = await to_thread.run_sync(pycaret.anomaly.create_model, **params.dict())
    return result

@app.post("/setup/classification/")
async def setup_classification(setup_params: ClassificationSetup):
    """Setup classification models with the provided parameters."""
    result = await to_thread.run_sync(pycaret.classification.setup, **setup_params.dict())
    return result

@app.post("/train/classification/")
async def train_classification(params: ClassificationParams):
    """Train classification models with the provided parameters."""
    result = await to_thread.run_sync(pycaret.classification.create_model, **params.dict())
    return result
