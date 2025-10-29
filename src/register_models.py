import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Connect to your local MLflow server
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("iris_ab_test")

# Load dataset
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train & register Version A
with mlflow.start_run(run_name="iris_model_A"):
    model_A = RandomForestClassifier(n_estimators=10, random_state=42)
    model_A.fit(X_train, y_train)
    mlflow.sklearn.log_model(model_A, "model", registered_model_name="iris_classifier")

# Train & register Version B
with mlflow.start_run(run_name="iris_model_B"):
    model_B = RandomForestClassifier(n_estimators=20, random_state=99)
    model_B.fit(X_train, y_train)
    mlflow.sklearn.log_model(model_B, "model", registered_model_name="iris_classifier")

print("✅ Two versions of 'iris_classifier' have been registered in MLflow.")