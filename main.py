from fastapi import FastAPI
from routers import anomaly_detection
from routers import classification
from routers import clustering
from routers import regression
from routers import time_series
import pycaret

app = FastAPI()

app.include_router(anomaly_detection.router)
app.include_router(classification.router)
app.include_router(clustering.router)
app.include_router(regression.router)
app.include_router(time_series.router)


@app.get("/")
async def root():
    return {"pycaret_version": pycaret.__version__}
