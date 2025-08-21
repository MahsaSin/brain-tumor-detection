from fastapi.testclient import TestClient

from inference.app import app  # assuming your FastAPI app is in main.py

client = TestClient(app)


def test_predict_success(mocker):
    # Mock pred.predict to return a fake detection result
    mocker.patch("inference.predict")

    response = client.post("/predict", json={"img_path": "sample-imgs/img1.jpg"})

    assert response.status_code == 200
    data = response.json()

    assert "predictions" in data
    assert data["predictions"][0]["class_name"] == "label1"
