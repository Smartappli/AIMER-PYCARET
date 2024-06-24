from enum import Enum
from numpy import ndarray
from scipy.sparse import spmatrix
from pandas import DataFrame, Series
from typing import Optional, Union, Callable, Dict, List, Any
from anyio import to_thread
from pycaret.loggers.base_logger import BaseLogger
import numpy
import pycaret
from fastapi import FastAPI
from pydantic import BaseModel
from logging import Logger
from joblib import Memory


app = FastAPI()


class ModelType(str, Enum):
    anomaly_detection = "anomaly detection"
    classification = "classification"
    clustering = "clustering"
    regression = "regression"
    time_series = "time series"


class ModelAnomalyDetection(str, Enum):
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
    log_experiment: bool | str | BaseLogger | List[str |  BaseLogger] = False
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
    model: str | Any
    fraction: float = 0.05
    verbose: bool = True
    fit_kwargs: Optional[dict] = None
    experiment_custom_tags: Optional[Dict[str, Any]] = None


class ModelClassification(str, Enum):
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
    data: Optional[dict | list | tuple | ndarray | spmatrix | DataFrame] = None
    data_func: Optional[
        Callable[[], dict | list | tuple | ndarray | spmatrix | DataFrame]
    ] = None
    target: int | str | list | tuple | ndarray | Series = -1
    index: bool | int | str | list | tuple | ndarray | Series = True
    train_size: float = 0.7
    test_data: Optional[
        dict | list | tuple | ndarray |  spmatrix | DataFrame
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
    engine: Optional[Dict[str, str]] = None
    verbose: bool = True
    memory: bool | str | Memory = True
    profile: bool = False
    profile_kwargs: Optional[Dict[str, Any]] = None


class ClassificationParams(BaseModel):
    estimator: str | Any
    fold: Optional[int | Any] = None
    round: int = 4
    cross_validation: bool = True
    fit_kwargs: Optional[dict] = None
    groups: Optional[str | Any] = None
    experiment_custom_tags: Optional[Dict[str, Any]] = None
    probability_threshold: Optional[float] = None
    engine: Optional[str] = None
    verbose: bool = True
    return_train_score: bool = False


class ModelClustering(str, Enum):
    kmeans = "K-Means Clustering"
    ap = "Affinity Propagation"
    meanshift = "Mean shift Clustering"
    sc = "Spectral Clustering"
    hclust = "Agglomerative Clustering"
    dbscan = "Density-Based Spatial Clustering"
    optics = "OPTICS Clustering"
    birch = "Birch Clustering"
    kmodes = "K-Modes Clustering"


class ClusteringSetup(BaseModel):
    data: Optional[dict | list | tuple | ndarray | spmatrix | DataFrame] = None
    data_func: Optional[Callable[[], dict | list | tuple | ndarray | spmatrix | DataFrame]] = None
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
    pca_components: Optional[int | float  str] = None
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


class ModelRegression(str, Enum):
    lr = "Linear Regression"
    lasso = "Lasso Regression"
    ridge = "Ridge Regression"
    en = "Elastic Net"
    lar = "Least Angle Regression"
    llar = "Lasso Least Angle Regression"
    omp = "Orthogonal Matching Pursuit"
    br = "Bayesian Ridge"
    ard = "Automatic Relevance Determination"
    par = "Passive Aggressive Regressor"
    ransac = "Random Sample Consensus"
    tr = "TheilSen Regressor"
    huber = "Huber Regressor"
    kr = "Kernel Ridge"
    svm = "Support Vector Regression"
    knn = "K Neighbors Regressor"
    dt = "Decision Tree Regressor"
    rf = "Random Forest Regressor"
    et = "Extra Trees Regressor"
    ada = "AdaBoost Regressor"
    gbr = "Gradient Boosting Regressor"
    mlp = "MLP Regressor"
    xgboost = "Extreme Gradient Boosting"
    lightgbm = "Light Gradient Boosting Machine"
    catboost = "CatBoost Regressor"


class RegressionSetup(BaseModel):
    data: Optional[dict | list | tuple | ndarray | spmatrix | DataFrame] = None
    data_func: Optional[Callable[[], dict | list | tuple | ndarray | spmatrix | DataFrame]] = None
    target: int | str | list | tuple | ndarray | Series = -1
    index: bool | int | str | list | tuple | ndarray | Series = True
    train_size: float = 0.7
    test_data: Optional[dict | list | tuple | ndarray | spmatrix | DataFrame] = None
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
    log_experiment: Union[
        bool, str, BaseLogger, List[Union[str, BaseLogger]]
    ] = False
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
    estimator: Union[str, Any]
    fold: Optional[Union[int, Any]] = None
    round: int = 4
    cross_validation: bool = True
    fit_kwargs: Optional[dict] = None
    groups: Optional[Union[str, Any]] = None
    experiment_custom_tags: Optional[Dict[str, Any]] = None
    engine: Optional[str] = None
    verbose: bool = True
    return_train_score: bool = False


class ModelTimeSeries(str, Enum):
    naive = "Naive Forecaster"
    grand_means = "Grand Means Forecaster"
    snaiv = "Seasonal Naive Forecaster"
    polytrend = "Polynomial Trend Forecaster"
    arima = "ARIMA family of models (ARIMA, SARIMA, SARIMAX)"
    auto_arima = "Auto ARIMA"
    exp_smooth = "Exponential Smoothing"
    stlf = "STL Forecaster"
    croston = "Croston Forecaster"
    ets = "ETS"
    theta = "Theta Forecaster"
    tbats = "TBATS"
    bats = "BATS"
    prophet = "Prophet Forecaster"
    lr_cds_dt = "Linear w/ Cond. Deseasonalize & Detrending"
    en_cds_dt = "Elastic Net w/ Cond. Deseasonalize & Detrending"
    ridge_cds_dt = "Ridge w/ Cond. Deseasonalize & Detrending"
    lasso_cds_dt = "Lasso w/ Cond. Deseasonalize & Detrending"
    llar_cds_dt = (
        "Lasso Least Angular Regressor w/ Cond. Deseasonalize & Detrending"
    )
    br_cds_dt = (
        "Bayesian Ridge w/ Cond. Deseasonalize & Deseasonalize & Detrending"
    )
    huber_cds_dt = "Huber w/ Cond. Deseasonalize & Detrending"
    omp_cds_dt = (
        "Orthogonal Matching Pursuit w/ Cond. Deseasonalize & Detrending"
    )
    knn_cds_d = "K Neighbors w/ Cond. Deseasonalize & Detrending"
    dt_cds_dt = "Decision Tree w/ Cond. Deseasonalize & Detrending"
    rf_cds_dt = "Random Forest w/ Cond. Deseasonalize & Detrending"
    et_cds_dt = "Extra Trees w/ Cond. Deseasonalize & Detrending"
    gbr_cds_dt = "Gradient Boosting w/ Cond. Deseasonalize & Detrending"
    ada_cds_dt = "AdaBoost w/ Cond. Deseasonalize & Detrending"
    lightgbm_cds_dt = (
        "Light Gradient Boosting w/ Cond. Deseasonalize & Detrending"
    )
    catboost_cds_dt = "CatBoost w/ Cond. Deseasonalize & Detrending"


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
    fh: Optional[Union[List[int], int, ndarray, ForecastingHorizon]] = 1
    hyperparameter_split: str = "all"
    seasonal_period: Optional[Union[List[Union[int, str]], int, str]] = None
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


def train_anomaly_detection_model(params: AnomalyDetectionParams):
    return {"model_type": "anomaly_detection", "params": params.dict()}


def train_classification_model(params: ClassificationParams):
    return {"model_type": "classification", "params": params.dict()}


def train_clustering_model(params: ClusteringParams):
    return {"model_type": "clustering", "params": params.dict()}


def train_regression_model(params: RegressionParams):
    return {"model_type": "regression", "params": params.dict()}


def train_time_series_model(params: TimeSeriesParams):
    return {"model_type": "time_series", "params": params.dict()}


@app.get("/")
async def root():
    return {"pycaret_version": pycaret.__version__}


@app.get("/model/{model_type}")
async def get_type(model_type: ModelType):
    if model_type == ModelType.anomaly_detection:
        return {model.name: model.value for model in ModelAnomalyDetection}
    elif model_type == ModelType.classification:
        return {model.name: model.value for model in ModelClassification}
    elif model_type == ModelType.clustering:
        return {model.name: model.value for model in ModelClustering}
    elif model_type == ModelType.regression:
        return {model.name: model.value for model in ModelRegression}
    elif model_type == ModelType.time_series:
        return {model.name: model.value for model in ModelTimeSeries}
    else:
        return {"error": "Model Type Unknown"}


@app.get("/type/{model_name}")
async def get_model(model_name: ModelType):
    return {"model_name": model_name}


@app.post("/model/{model_type}")
async def create_model(model_type: ModelType, params: ClassificationParams):
    if model_type == ModelType.anomaly_detection:
        result = await to_thread.run_sync(train_anomaly_detection_model, params)
        return result
    elif model_type == ModelType.classification:
        result = await to_thread.run_sync(train_classification_model, params)
        return result
    elif model_type == ModelType.clustering:
        result = await to_thread.run_sync(train_clustering_model, params)
        return rsult
    elif model_type == ModelType.regression:
        result = await to_thread.run_sync(train_regression_model, params)
        return result
    elif model_type == ModelType.time_series:
        result = await to_thread.run_sync(train_time_serues_model, params)
        return result
    else:
        return {"error": "Model Type Not Supported for Training"}


@app.post("/anomaly-detection")
async def anomaly_detection(params: AnomalyDetectionParams):
    return train_anomaly_detection_model(params)


@app.post("/classification")
async def classification(params: ClassificationParams):
    return train_classification_model(params)


@app.post("/clustering")
async def clustering(params: ClusteringParams):
    return train_clustering_model(params)


@app.post("/regression")
async def regression(params: RegressionParams):
    return train_regression_model(params)


@app.post("/time-series")
async def time_series(params: TimeSeriesParams):
    return train_time_series_model(params)
