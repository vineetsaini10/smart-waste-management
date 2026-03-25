# AI/ML Backend for Waste Management System

This module provides the core artificial intelligence capabilities for the Waste Management System, exposing FastAPI REST endpoints. The system is designed to be highly modular and production-ready.

## Features
1. **Waste Image Classification**: Uses MobileNetV2 (Computer Vision) via PyTorch predicting 4 subclasses.
2. **Geo-Spatial Hotspot Detection**: Uses DBSCAN clustering logic (Geo AI) via Scikit-Learn to detect groupings.
3. **Predictive Analytics**: Uses RandomForest regression for generic time-series forecasting.

## Setup Requirements

- Python 3.10+
- `pip` package manager

## How to Run Locally

1. **Navigate to the project root directory** (one level above this `aiml` folder):
   ```bash
   cd ..
   ```

2. **Install dependencies**:
   ```bash
   pip install -r aiml/requirements.txt
   ```

3. **Run the FastAPI server**:
   ```bash
   uvicorn aiml.main:app --host 0.0.0.0 --port 8000 --reload
   ```

4. **View Swagger Documentation**:
   Open a browser and navigate to `http://localhost:8000/docs` to visualize and interact with the endpoints.

## How to Run via Docker

1. **Navigate to the `aiml` directory**:
   ```bash
   cd aiml
   ```

2. **Build the Docker Image**:
   ```bash
   docker build -t aiml-backend .
   ```

3. **Run the Docker Container**:
   ```bash
   docker run -d -p 8000:8000 aiml-backend
   ```
