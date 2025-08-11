from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from inference.predict import Predictor

app = FastAPI()
pred = Predictor()


@app.post("/predict")
async def predict(img_path):
    try:
        class_boxes = pred.predict(img_path=img_path)

        return JSONResponse({"predictions": class_boxes})

    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Invalid image or prediction error: {e}"
        )
