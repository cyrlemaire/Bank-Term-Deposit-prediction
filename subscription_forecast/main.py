from fastapi import FastAPI, HTTPException, UploadFile
from pydantic import BaseModel
import pickle
import os
import uvicorn
import pandas as pd
from io import BytesIO


app =FastAPI()


def load_model():
    with open("finalized_model.sav", 'rb') as f:
        return pickle.load(f)


model_pipeline = load_model()

# Define the response JSON
class Prediction(BaseModel):
    predictions: int


@app.post("/predict", response_model=Prediction)
def prediction(input_csv: str):
    if not os.path.exists(input_csv):
        raise HTTPException(status_code=400, detail="File provided does not exist.")

    result_subscription = model_pipeline(input_csv, run_type='api')
    return {"predictions": result_subscription}


if __name__ == "__main__":
    uvicorn.run("main.app", host="0.0.0.0", port=5000)
