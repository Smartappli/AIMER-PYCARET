from enum import Enum
from typing import Any, Dict, Optional

import pycaret
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


class ModelType(str, Enum):
    classification = "classification"
    regression = "regression"
    clustering = "clustering"
    anomaly_detection = "anomaly_detection"


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


@app.get("/")
async def root():
    return {"pycaret_version": pycaret.__version__}


@app.get("/model/{model_type}")
async def get_type(model_type: ModelType):
    if model_type == ModelType.classification:
        return {model.name: model.value for model in ModelClassification}
    elif model_type == ModelType.regression:
        return {model.name: model.value for model in ModelRegression}
    elif model_type == ModelType.clustering:
        return {model.name: model.value for model in ModelClustering}
    elif model_type == ModelType.anomaly_detection:
        return {model.name: model.value for model in ModelAnomalyDetection}
    else:
        return {"error": "Model Type Unknown"}


@app.get("/type/{model_name}")
async def get_model(model_name: ModelType):
    return {"model_name": model_name}


@app.post("/model/{model_type")
async def create_model(model_type: ModelType):
    if model_type == ModelType.classification:
        return {"model_name": model_type.name}
