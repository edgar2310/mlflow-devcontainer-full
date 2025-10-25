from src.train import train

def test_training_runs():
    metrics = train(run_name="pytest_run", test_size=0.3, random_state=0, C=0.5, max_iter=200)
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["roc_auc"] <= 1.0
