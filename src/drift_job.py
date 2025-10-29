import os
import pandas as pd
import mlflow
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

REF_PATH = "data/reference_iris.csv"
CUR_PATH = "data/live/live_A.csv"
OUT_DIR = "data/drift_reports"
os.makedirs(OUT_DIR, exist_ok=True)

def main():
    if not os.path.exists(REF_PATH) or not os.path.exists(CUR_PATH):
        raise FileNotFoundError("Missing reference or current data!")

    df_ref = pd.read_csv(REF_PATH)
    df_cur = pd.read_csv(
        CUR_PATH,
        names=df_ref.columns,
        header=0 if os.stat(CUR_PATH).st_size else None,
    )

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=df_ref, current_data=df_cur)

    html_path = os.path.join(OUT_DIR, "drift_report.html")
    report.save_html(html_path)
    print(f"✅ Drift report saved to {html_path}")

    # ---- Parse drift value safely across Evidently versions ----
    drift_share = 0.0
    try:
        result = report.as_dict()
        metrics = result.get("metrics", [])
        if metrics:
            res = metrics[0].get("result", {})
            # v0.5.x and older
            if isinstance(res, dict):
                if "dataset_drift" in res and isinstance(res["dataset_drift"], dict):
                    drift_share = res["dataset_drift"].get("share_drifted_features", 0.0)
                elif "share_drifted_features" in res:
                    drift_share = res["share_drifted_features"]
                elif "drift_share" in res:
                    drift_share = res["drift_share"]
                elif isinstance(res.get("dataset_drift"), bool):
                    drift_share = 1.0 if res["dataset_drift"] else 0.0
            elif isinstance(res, bool):
                drift_share = 1.0 if res else 0.0
    except Exception as e:
        print("⚠️ Could not parse drift metric:", e)
        drift_share = 0.0

    print(f"📊 Share of drifted features: {drift_share}")

    # ---- Log to MLflow ----
    if not os.getenv("MLFLOW_TRACKING_URI"):
        os.environ["MLFLOW_TRACKING_URI"] = "file:" + os.path.abspath("mlruns")

    with mlflow.start_run(run_name="drift_monitoring") as run:
        mlflow.log_metric("share_drifted_features", float(drift_share))
        mlflow.log_artifact(html_path, artifact_path="drift_report")

    print("✅ Drift metrics logged to MLflow")

if __name__ == "__main__":
    main()