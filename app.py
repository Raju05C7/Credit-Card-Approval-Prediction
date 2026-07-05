from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained Random Forest model
with open("credit_card_model.pkl", "rb") as file:
    model = pickle.load(file)


# -------------------- HOME PAGE -------------------- #
@app.route("/")
def home():
    return render_template("home.html")


# -------------------- PREDICTION -------------------- #
@app.route("/predict", methods=["GET", "POST"])
def predict():

    if request.method == "POST":

        try:

            # Read Time
            time = float(request.form["Time"])

            # Read V1 to V28
            features = [time]

            for i in range(1, 29):
                value = float(request.form[f"V{i}"])
                features.append(value)

            # Read Amount
            amount = float(request.form["Amount"])
            features.append(amount)

            # Convert to NumPy array
            features = np.array(features).reshape(1, -1)

            # Prediction
            prediction = model.predict(features)[0]

            # Confidence Score
            probability = model.predict_proba(features)
            confidence = round(np.max(probability) * 100, 2)

            if prediction == 0:
                result = "Legitimate Transaction"
                color = "success"
                icon = "✅"
            else:
                result = "Fraudulent Transaction"
                color = "danger"
                icon = "⚠️"

            return render_template(
                "result.html",
                prediction=result,
                confidence=confidence,
                color=color,
                icon=icon
            )

        except Exception as e:
            return f"<h2>Error:</h2><br>{e}"

    return render_template("index.html")


# -------------------- RUN APP -------------------- #
if __name__ == "__main__":
    app.run(debug=True)