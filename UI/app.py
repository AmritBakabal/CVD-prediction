#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load the trained model from file
model = joblib.load('trained_model.sav')

# Define the route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define the route for the prediction page
@app.route('/predict', methods=['POST'])
def predict():
    print(request.form)
    # Get input values from form
    age = int(request.form['age'])
    sex = int(request.form['sex'])
    cigsPerDay = float(request.form['cigsPerDay'])
    BPMeds = float(request.form['BPMeds'])
    prevalentStroke = int(request.form['prevalentStroke'])
    totChol = float(request.form['totChol'])
    BMI = float(request.form['BMI'])
    heartRate = float(request.form['heartRate'])
    MAP = float(request.form['MAP'])
    diabetes_grade = int(request.form['diabetes_grade'])

    # Create dataframe with input values
    data = pd.DataFrame({'age': [age],
                         'sex': [sex],
                         'cigsPerDay': [cigsPerDay],
                         'BPMeds': [BPMeds],
                         'prevalentStroke': [prevalentStroke],
                         'totChol': [totChol],
                         'BMI': [BMI],
                         'heartRate': [heartRate],
                         'MAP': [MAP],
                         'diabetes_grade': [diabetes_grade]})

    # Make prediction and display result
    result = model.predict(data)[0]
    if result == 0:
        prediction = 'No Risk of heart disease'
    else:
        prediction = 'Risk of heart disease'

    # Render the prediction page with the result
    return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)

