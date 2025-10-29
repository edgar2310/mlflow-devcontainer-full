import argparse
import os
from typing import Tuple

import mlflow
import mlflow.sklearn
from mlflow import MlflowClient

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from mlflow.models.signature import infer_signature


def get_data(test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    data = load_iris(as_frame=True)
    X = data.frame.drop(columns=["target"])
    y = data.frame["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, y_train, X_test, y_test


def plot_confusion(cm: np.ndarray, labels: list, out_path: str, title: str):
    import numpy as np
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, interpolation='nearest')
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]),
           xticklabels=labels, yticklabels=labels, ylabel='True label', xlabel='Predicted label',
           title=title)
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def train_and_log(model_name: str, model, X_train, y_train, X_test, y_test, run_name: str):
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", model)])
    with mlflow.start_run(run_name=run_name) as run:
        # Fit
        pipe.fit(X_train, y_train)

        # Predict
        y_pred = pipe.predict(X_test)

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="macro")

        mlflow.log_metric("accuracy", float(acc))
        mlflow.log_metric("f1_macro", float(f1))

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        os.makedirs("artifacts", exist_ok=True)
        cm_path = os.path.join("artifacts", f"cm_{run_name}.png")
        plot_confusion(cm, labels=["setosa","versicolor","virginica"], out_path=cm_path, title=f"CM {run_name}")
        mlflow.log_artifact(cm_path, artifact_path="plots")

        # Log model with signature
        signature = infer_signature(X_test, pipe.predict(X_test))
        mlflow.sklearn.log_model(pipe, artifact_path="model", signature=signature, input_example=X_test.head(3))

        # Return run info
        return run.info.run_id


def register_version(registered_name: str, run_id: str) -> str:
    client = MlflowClient()
    model_uri = f"runs:/{run_id}/model"
    # Ensure registered model exists
    try:
        client.create_registered_model(registered_name)
    except Exception:
        pass
    mv = client.create_model_version(registered_name, model_uri, run_id)
    return mv.version


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--run-name", type=str, default="seed_models")
    parser.add_argument("--no-register", action="store_true", help="Skip model registry steps")
    args = parser.parse_args()

    # Default to local file tracking store if not set
    if not os.getenv("MLFLOW_TRACKING_URI"):
        os.environ["MLFLOW_TRACKING_URI"] = "file:" + os.path.abspath("mlruns")

    X_train, y_train, X_test, y_test = get_data(test_size=args.test_size, random_state=args.random_state)

    # Train two different models (A & B)
    run_id_a = train_and_log(
        model_name="LogisticRegression",
        model=LogisticRegression(C=0.5, max_iter=1000, multi_class="auto"),
        X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, run_name="iris_A_logreg_C0.5"
    )
    run_id_b = train_and_log(
        model_name="RandomForest",
        model=RandomForestClassifier(n_estimators=120, max_depth=5, random_state=42),
        X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, run_name="iris_B_rf_depth5"
    )

    print(f"Run A: {run_id_a}")
    print(f"Run B: {run_id_b}")

    if not args.no_register:
        name = "iris_clf"
        v_a = register_version(name, run_id_a)
        v_b = register_version(name, run_id_b)
        print(f"Registered versions: A -> v{v_a}, B -> v{v_b}")
        print("Tip: Start MLflow server with a DB backend to use the full Registry UI:\n"
              "mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns "
              "--host 0.0.0.0 --port 5000")


if __name__ == "__main__":
    main()
