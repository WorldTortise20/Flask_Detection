from flask import Flask, jsonify, request
from Alphabet import prediction

app = Flask(__name__)

@app.route("/predict-alphabet", methods = ["POST"])

def predict_data():
    image = request.files.get("alphabet")
    pred = prediction(image)
    return jsonify ({"prediction": pred}), 200

if __name__ == "__main__":
    app.run(debug = True)