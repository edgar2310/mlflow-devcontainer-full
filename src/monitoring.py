from prometheus_client import Counter, Histogram
import time

REQUEST_COUNT = Counter(
    "prediction_requests_total",
    "Total prediction requests",
    ["model_version"]
)

REQUEST_LATENCY = Histogram(
    "prediction_request_latency_seconds",
    "Latency for model predictions",
    ["model_version"]
)

def track_request(model_version: str):
    """Decorator to measure latency and increment counters."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            latency = time.time() - start
            REQUEST_COUNT.labels(model_version=model_version).inc()
            REQUEST_LATENCY.labels(model_version=model_version).observe(latency)
            return result
        return wrapper
    return decorator