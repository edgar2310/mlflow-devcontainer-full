import os
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

def main():
    data = load_iris(as_frame=True)
    X = data.frame.drop(columns=["target"])
    y = data.frame["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    os.makedirs("data", exist_ok=True)
    ref_path = os.path.join("data", "reference_iris.csv")
    cur_path = os.path.join("data", "current_iris.csv")  # placeholder updated by drift jobs

    df_ref = X_train.copy()
    df_ref["target"] = y_train.values
    df_ref.to_csv(ref_path, index=False)

    df_cur = X_test.copy()
    df_cur["target"] = y_test.values
    df_cur.to_csv(cur_path, index=False)

    print(f"Saved reference baseline to {ref_path} and initial current to {cur_path}")

if __name__ == "__main__":
    main()
