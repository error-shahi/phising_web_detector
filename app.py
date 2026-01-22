from flask import Flask, render_template, request
import joblib
from features import extract_features

app = Flask(__name__)

# Load the trained model
model = joblib.load("phishing_model.pkl")

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/scan", methods=["POST"])
def scan():
    url = request.form["url"]
    features = extract_features(url)
    prediction = model.predict([features])[0]
    result = "⚠️ PHISHING WEBSITE" if prediction == 1 else "✅ SAFE WEBSITE"
    return render_template("result.html", result=result, url=url)

if __name__ == "__main__":
    # Render uses gunicorn, so this block is optional in production
    app.run(debug=True)
