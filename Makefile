# Convenience commands
.PHONY: train ui test

train:
	python -m src.train --run-name local_dev_run

ui:
	mlflow server --backend-store-uri sqlite:///mlflow.db \
	              --default-artifact-root ./mlruns \
	              --host 0.0.0.0 --port 5000

test:
	pytest -q
