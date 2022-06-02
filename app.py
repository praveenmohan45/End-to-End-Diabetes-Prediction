from flask import Flask, render_template, request
from joblib import load

import numpy as np

from flask import Flask

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/about")
def about():
    return render_template("About.html")


@app.route("/diabetes")
def diabetes():
    return render_template("Diabetes.html")


@app.route("/diabetes_predict", methods=["POST"])
def diabetes_predict():
    model = load('Diabetes Detection Notebook/diabeteseModel.joblib')

    # Taking the Inputs from the Form using POST Method and converting to float
    if request.method == 'POST':
        Pregnancies = float(request.form.get('Pregnancies'))
        Glucose = float(request.form.get('Glucose'))
        BloodPressure = float(request.form.get('BloodPressure'))
        SkinThickness = float(request.form.get('SkinThickness'))
        Insulin = float(request.form.get('Insulin'))
        BMI = float(request.form.get('BMI'))
        DiabetesPedigreeFunction = float(request.form.get('DiabetesPedigreeFunction'))
        Age = float(request.form.get('Age'))
        # print(Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age)

        # Prediction features
        features = np.array(
            [[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])

        # Predicting the Resulting using model and converting to int
        results_predict = int(model.predict(features))

        # Printing the results for understanding
        print(results_predict)

        # According to results redirecting to results page
        if results_predict:
            prediction = "There are chances of Diabetes ! Consult your doctor Soon."
            # print("There are chances of Diabetes ! Consult your doctor Soon.")

        else:
            prediction = "No need to fear. You have no dangerous symptoms of the Diabetes. For Better Understanding " \
                         "you can consult your doctor! "
            # print("No diabetes Chances")

    return render_template("result.html", prediction_text=prediction)



app.run(debug=True)
