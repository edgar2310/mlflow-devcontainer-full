import time
import requests
import numpy as np
import json

API_URL = "http://localhost:8000/predict"

def generate_sample():
    """Generate a random sample for the Iris model (4 features)."""
    # Using random floats as mock input for sepal/petal features
    return {
        "sepal_length": float(np.random.uniform(4.5, 7.5)),
        "sepal_width": float(np.random.uniform(2.0, 4.5)),
        "petal_length": float(np.random.uniform(1.0, 6.5)),
        "petal_width": float(np.random.uniform(0.1, 2.5))
    }

def main():
    print("🚀 Sending inference requests to FastAPI A/B service...")
    while True:
        sample = generate_sample()
        try:
            # Send JSON body
            resp = requests.post(API_URL, json=sample, timeout=2)
            if resp.status_code == 200:
                print(resp.json())
            else:
                print(f"Error: {resp.status_code} -> {resp.text}")
        except Exception as e:
            print(f"Error: {e}")
        time.sleep(1)  # wait 1s between requests

if __name__ == "__main__":
    main()