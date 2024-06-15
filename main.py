from enum import Enum
from fastapi import FastAPI
import pycaret

app = FastAPI()

class ModelType(str, Enum):
    classification = "classification"
    regression = "regression"
    clustering = "clustering"


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


@app.get("/")
async def root():
    return {pycaret.__version__}

@app.get("/type/{model_type}")
async def get_type(model_type: ModelType):
    match (model_type)
        case ModelType.classification:
            return "blabla"
        caee ModelType.regression:
            return "ok"
        case ModelType.clustering:
            return "coucou"
        case _:
            return "Model Type Unkown"

@app.get("/models/{model_name}")
async def get_model(model_name: ModelName):
    return model_name
