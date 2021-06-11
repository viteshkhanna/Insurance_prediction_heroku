from logging import debug
from flask import Flask , render_template , request
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load('Insuranceprediction.pkl')
@app.route('/home')
def Welcome():
    return render_template('base.html')

@app.route('/predict', methods = ['GET',POST'])
def predict():
    age = request.form.get('age')
    sex = request.form.get('sex')
    bmi = request.form.get('bmi')
    children = request.form.get('children')
    smoker = request.form.get('smoker')
    region = request.form.get('region')


    prediction  = model.predict([[age,sex,bmi,children,smoker,region]])
    output = round(prediction[0],2)

    return render_template('base.html' , prediction_text = f"Insurance Expense will be ${output}")

app.run(debug=True)
