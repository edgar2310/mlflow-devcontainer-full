from fastapi import FastAPI
import mlflow.sklearn
import pandas as pd
import os

# load the latest model (from local mlruns)
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "file:" + os.path.abspath("mlruns"))
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# load the most recent run's model
runs = mlflow.search_runs(order_by=["start_time DESC"], max_results=1)
run_id = runs.loc[0, "run_id"]
model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")

app = FastAPI(title="MLflow Breast Cancer Classifier API")

@app.get("/")
def home():
    return {"message": "MLflow model is ready for inference!"}

@app.post("/predict")
def predict(data: dict):
    """Example input:
    {
      "mean radius": 14.5,
      "mean texture": 19.1,
      "mean perimeter": 94.3,
      ...
    }
    """
    df = pd.DataFrame([data])
    y_pred = model.predict(df)[0]
    y_prob = model.predict_proba(df)[0, 1]
    return {"prediction": int(y_pred), "probability": float(y_prob)}