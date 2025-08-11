from ultralytics import YOLO

from inference.conf import MODEL_PATH


class Predictor:
    def __init__(self, model_path=MODEL_PATH):
        self.model = YOLO(model_path)

    def predict(self, img_path):
        results = self.model(img_path, verbose=False)
        boxes = results[0].boxes
        class_names = self.model.names

        class_boxes = {}
        for box in boxes:
            cls_id = int(box.cls[0])
            cls_name = class_names[cls_id]
            xywh = box.xywh[0].tolist()
            class_boxes.setdefault(cls_name, []).append(xywh)

        return class_boxes


if __name__ == "__main__":
    pred = Predictor()
    output = pred.predict(
        "volume_1_slice_100_jpg.rf.b3e873b1f4d54ee6c59a12a7b5e3bdcb.jpg"
    )
    print(output)
