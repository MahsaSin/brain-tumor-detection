from pydantic import BaseModel

class Detection(BaseModel):
    class_name: str
    confidence: float
    bbox: list[float]  # [x_min, y_min, x_max, y_max]

class Result(BaseModel):
    predictions: list[Detection]


class InputData(BaseModel):
    img_path: str 
