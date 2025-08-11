import os

import mlflow
from dotenv import load_dotenv
from omegaconf import OmegaConf
from ultralytics import YOLO

load_dotenv()


def train(conf_path="train/config.yaml"):
    conf = OmegaConf.load(conf_path)
    print(conf)

    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI")

    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(conf.experiment_name)
    mlflow.pytorch.autolog()

    model = YOLO(conf.model_path)

    results = model.train(data=conf.data_path, epochs=conf.epochs, imgsz=conf.imgsz)


if __name__ == "__main__":
    train()
