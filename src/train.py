import json
import os
import sys
import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Quality gate threshold (set to 0.999 to trigger deliberate failure)
ACCURACY_THRESHOLD = 1.01


def train_and_evaluate(threshold=ACCURACY_THRESHOLD):
    data = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print(f"Model Accuracy: {accuracy:.4f}")
    print(f"Required Threshold: {threshold:.4f}")

    os.makedirs("artifacts", exist_ok=True)
    joblib.dump(model, "artifacts/model.joblib")

    with open("artifacts/metrics.json", "w") as f:
        json.dump({"accuracy": accuracy, "threshold": threshold}, f, indent=2)

    if accuracy < threshold:
        print(
            f"QUALITY GATE FAILED: Accuracy {accuracy:.4f} is below {threshold:.4f}"
        )
        sys.exit(1)

    print("QUALITY GATE PASSED!")
    return accuracy


if __name__ == "__main__":
    train_and_evaluate()
