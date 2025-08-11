# Brain Tumor Detection Project

A machine learning project detecting brain tumors from medical images.  
Built with **Python 3.12** and , structured for readability, maintainability, easy extension to an API, and Dockerized.

Simple wrapper around Ultralytics YOLO to detect brain tumors in MRI images.
Returns bounding boxes grouped by class.

---

## Project Structure

notebooks/       → Exploration & experiments  
src/             → Core logic (processing, inference)  
models/          → (Ignored) Trained model files  
.gitignore       → Ignore models & artifacts  
.python-version  → Python 3.12  
pyproject.toml   → Metadata & dependencies  
uv.lock          → Locked deps for reproducibility  
README.md        → You’re here



---

## Overview

This project aims to detect the presence of brain tumors in medical imaging data (e.g., MRI scans) using machine learning. It's thoughtfully structured so you can:

- Prototype and explore in **notebooks/**  
- Refine and migrate logic into **src/** for production-quality code  
- Add an HTTP-based inference **API** (e.g., using FastAPI)  
- Train or update models with minimal friction

---

## Getting Started

### Clone the repository

```bash
git clone https://github.com/MahsaSin/brain-tumor-detection.git
cd brain-tumor-detection
```

## Running with Docker

The project includes a `Dockerfile` based on Python 3.12 slim, pre-configured with dependencies like `ffmpeg` and `uv`.

**Build the Docker image:**
```bash
docker build -t my-inference-app .
```

**Run the container:**
```bash
docker run -p 8000:8000 my-inference-app
```

**Once running, the API will be accessible at:**
```bash
http://localhost:
```

