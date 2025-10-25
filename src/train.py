import argparse
import os
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
)

def get_data():
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="target")
    return X, y

def build_pipeline(C: float = 1.0, max_iter: int = 1000, random_state: int = 42):
    pipe = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            C=C,
            solver="lbfgs",
            max_iter=max_iter,
            random_state=random_state
        ))
    ])
    return pipe

def plot_confusion_matrix(cm: np.ndarray, labels: list, out_path: str):
    fig, ax = plt.subplots(figsize=(4,4))
    im = ax.imshow(cm, interpolation='nearest')
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]),
           xticklabels=labels, yticklabels=labels, ylabel='True label', xlabel='Predicted label',
           title='Confusion Matrix')
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

def train(run_name: str = "local_run", test_size: float = 0.2, random_state: int = 42,
          C: float = 1.0, max_iter: int = 1000):
    # Ensure local file store default if env not set
    if not os.getenv("MLFLOW_TRACKING_URI"):
        os.environ["MLFLOW_TRACKING_URI"] = "file:" + os.path.abspath("mlruns")

    X, y = get_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

    model = build_pipeline(C=C, max_iter=max_iter, random_state=random_state)

    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("C", C)
        mlflow.log_param("max_iter", max_iter)
        mlflow.log_param("test_size", test_size)
        mlflow.log_param("random_state", random_state)

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred)),
            "recall": float(recall_score(y_test, y_pred)),
            "f1": float(f1_score(y_test, y_pred)),
            "roc_auc": float(roc_auc_score(y_test, y_prob)),
        }
        for k, v in metrics.items():
            mlflow.log_metric(k, v)

        cm = confusion_matrix(y_test, y_pred)
        os.makedirs("artifacts", exist_ok=True)
        cm_path = os.path.join("artifacts", "confusion_matrix.png")
        plot_confusion_matrix(cm, labels=["malignant", "benign"], out_path=cm_path)
        mlflow.log_artifact(cm_path, artifact_path="plots")

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            input_example=X_test.head(3)
        )

        return metrics

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run-name", type=str, default="local_run")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--C", type=float, default=1.0)
    p.add_argument("--max-iter", type=int, default=1000)
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train(run_name=args.run_name, test_size=args.test_size, random_state=args.random_state, C=args.C, max_iter=args.max_iter)
