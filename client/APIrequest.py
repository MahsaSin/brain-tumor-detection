import requests 

url = "http://127.0.0.1:8000/predict"
data = {"img_path": "sample-imgs/img1.jpg"}

response = requests.post(url, json=data)


if __name__ == "__main__":
    print("Status Code:", response.status_code)
    print("Response Body:", response.json())
