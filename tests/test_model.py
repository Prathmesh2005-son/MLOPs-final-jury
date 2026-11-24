import os
from src.train import train_and_evaluate


def test_training_pipeline():
    acc = train_and_evaluate(threshold=0.80)
    assert acc >= 0.80
    assert os.path.exists("artifacts/model.joblib")
    assert os.path.exists("artifacts/metrics.json")
