import sys
import mlflow
import mlflow.sklearn
import pandas as pd

def main(csv_path: str):
    print("Placeholder inference script. In a real project, register the model and load by name/stage.")
    print("For now, run training, find the run in MLflow UI, and use the artifact path to load the model.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.infer <csv_path>")
        sys.exit(1)
    main(sys.argv[1])
