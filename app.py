import numpy as np
import joblib
from flask import Flask, render_template, request

app = Flask(__name__)

# Load the trained model
model = joblib.load("fitness_model.pkl")

# You may want to update this mapping according to your LabelEncoder classes_
activity_mapping = {
    0: "Activity0",
    1: "Jogging",
    2: "Sitting",
    3: "Activity3",
    4: "Activity4",
    5: "Activity5"
}

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        try:
            x = float(request.form["x"])
            y = float(request.form["y"])
            z = float(request.form["z"])
            # For scaling, you should use the same scaler as in training. 
            # This example assumes input is already scaled or scaler is not saved.
            features = np.array([[x, y, z]])
            pred = model.predict(features)[0]
            activity = activity_mapping.get(pred, f"Label {pred}")
            prediction = f"Predicted activity: {activity}"
        except Exception as e:
            prediction = f"Error: {str(e)}"
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)