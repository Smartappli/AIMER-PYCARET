from enum import Enum
from fastapi import FastAPI
import pycaret

app = FastAPI()

class ModelType(str, Enum):
    classification = "classification"
    regression = "regression"
    clustering = "clustering"

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
