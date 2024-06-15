from fastapi import FastAPI
import pycaret

app = FastAPI()


@app.get("/")
async def root():
    return {pycaret.__version__}
