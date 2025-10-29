from fastapi import FastAPI, Request, Response
import random
import mlflow.pyfunc
import numpy as np
import pandas as pd
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST


mlflow.set_tracking_uri("http://localhost:5000")
# ------------------------------------------------------------
# Load two model versions from MLflow registry
# ------------------------------------------------------------
MODEL_A = mlflow.pyfunc.load_model("models:/iris_classifier/1")
MODEL_B = mlflow.pyfunc.load_model("models:/iris_classifier/2")

# ------------------------------------------------------------
# FastAPI + Prometheus setup
# ------------------------------------------------------------
app = FastAPI(title="Iris A/B Testing Service")

# Prometheus metrics
REQUEST_COUNT = Counter(
    "prediction_requests_total",
    "Total prediction requests",
    ["model_version"]
)
LATENCY = Histogram(
    "prediction_request_latency_seconds",
    "Latency for model predictions",
    ["model_version"]
)

# ------------------------------------------------------------
# Root endpoint
# ------------------------------------------------------------
@app.get("/")
def home():
    return {"message": "MLflow A/B test service is running 🚀"}

# ------------------------------------------------------------
# Proper Prometheus endpoint
# ------------------------------------------------------------
@app.get("/metrics")
def metrics():
    data = generate_latest()
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)

# ------------------------------------------------------------
# Inference endpoint
# ------------------------------------------------------------
@app.post("/predict")
async def predict(request: Request):
    try:
        payload = await request.json()
        df = pd.DataFrame([payload])

        # Randomly assign model A or B
        model_version = random.choice(["A", "B"])
        model = MODEL_A if model_version == "A" else MODEL_B

        with LATENCY.labels(model_version).time():
            preds = model.predict(df)

        REQUEST_COUNT.labels(model_version).inc()
        return {"model": model_version, "prediction": int(preds[0])}

    except Exception as e:
        return {"error": str(e)}

# ------------------------------------------------------------
# Startup message
# ------------------------------------------------------------
@app.on_event("startup")
def startup_event():
    print("✅ Iris A/B Test API Ready (http://0.0.0.0:8000)")