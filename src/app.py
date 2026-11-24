import os
import joblib
from flask import Flask, jsonify, request

app = Flask(__name__)
MODEL_PATH = os.getenv("MODEL_PATH", "artifacts/model.joblib")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy"})


@app.route("/predict", methods=["POST"])
def predict():
    model = joblib.load(MODEL_PATH)
    features = request.json.get("features")
    prediction = model.predict([features]).tolist()
    return jsonify({"prediction": prediction[0]})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
