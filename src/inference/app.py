from fastapi import FastAPI, HTTPException

from inference.predict import Predictor
from inference.schemas import Result, InputData

app = FastAPI()
pred = Predictor()


@app.post("/predict")
async def predict(input_data: InputData):
    try:
        detections = pred.predict(input_data.img_path)

        result = Result(
            predictions=detections
        )

        return  result

    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Invalid image or prediction error: {e}"
        )
