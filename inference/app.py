from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

with open("/model/model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json["features"]
    prediction = model.predict([data])

    return jsonify({
        "prediction": int(prediction[0])
    })

# NEW ENDPOINT
@app.route("/model-info", methods=["GET"])
def model_info():
    return jsonify({
        "model_type": str(type(model).__name__),
        "features_expected": getattr(model, "n_features_in_", None)
    })

@app.route("/")
def home():
    return "Inference running"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)