# MLflow Dev Container Starter (Complete)

End-to-end **classification** project using **MLflow**, **VS Code Dev Containers**, and **GitHub Actions**.

### Quickstart
```bash
# 1) Clone or unzip
cd mlflow-devcontainer-starter

# 2) Open in VS Code → Command Palette → "Reopen in Container"
#    (or the bottom-right prompt)

# 3) Run tests
make test

# 4) Train
make train

# 5) MLflow UI
make ui   # then open http://localhost:5000
```

### Notes
- Tracking URI defaults to a local **file store** at `./mlruns` (configured in Dev Container).
- Swap to a remote tracking server later by setting `MLFLOW_TRACKING_URI` and using a proper backend store.
# prueba CI
