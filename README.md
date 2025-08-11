# Brain Tumor Detection Project

A machine learning project detecting brain tumors from medical images.  
Built with **Python 3.12**, structured for readability, maintainability, and easy extension to an API.

---

## Project Structure

├── notebooks/ # Jupyter notebooks for exploration, experiments (primary content)
├── src/ # Core application code (processing, inference, etc.)
├── models/ # (Ignored) Trained model files and weights
├── .gitignore # Git exclude for models and other artifacts
├── .python-version # Python version specified (3.12)
├── pyproject.toml # Project metadata and dependency declarations
├── uv.lock # Locked dependencies for reproducible environments
└── README.md # Project documentation you're reading now


---

## Overview

This project aims to detect the presence of brain tumors in medical imaging data (e.g., MRI scans) using machine learning. It's thoughtfully structured so you can:

- Prototype and explore in **notebooks/**  
- Refine and migrate logic into **src/** for production-quality code  
- Add an HTTP-based inference **API** (e.g., using FastAPI)  
- Train or update models with minimal friction

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/MahsaSin/brain-tumor-detection.git
cd brain-tumor-detection
