from ultralytics import YOLO

from inference.conf import MODEL_PATH
from inference.schemas import Detection


class Predictor:
    def __init__(self, model_path=MODEL_PATH):
        self.model = YOLO(model_path)

    def predict(self, img_path):
        results = self.model(img_path, verbose=False)
        boxes = results[0].boxes
        class_names = self.model.names

        detections = []
        for box in boxes:
            cls_id = int(box.cls[0])
            cls_name = class_names[cls_id]
            xyxy = box.xyxy[0].tolist()

            det = Detection(
                class_name=cls_name,
                confidence=float(box.conf[0]),
                bbox=xyxy
            )
            detections.append(det)

        return detections


if __name__ == "__main__":
    pred = Predictor()
    output = pred.predict(
        "volume_1_slice_100_jpg.rf.b3e873b1f4d54ee6c59a12a7b5e3bdcb.jpg"
    )
    print(output)
