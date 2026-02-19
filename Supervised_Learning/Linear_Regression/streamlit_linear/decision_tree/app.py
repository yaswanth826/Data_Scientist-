from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        income = float(request.form["income"])
        credit_score = float(request.form["credit_score"])
        has_job = int(request.form["has_job"])

        features = np.array([[income, credit_score, has_job]])
        prediction = model.predict(features)[0]

        result = "✅ Loan Approved" if prediction == 1 else "❌ Loan Rejected"

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)