from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

# Load the model and scaler
model = joblib.load('model/loan_status_predictor.pkl')
scaler = joblib.load('model/vector.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect data from form
        user_input = {
            'Gender': int(request.form['gender']),
            'Married': int(request.form['married']),
            'Dependents': int(request.form['dependents']),
            'Education': int(request.form['education']),
            'Self_Employed': int(request.form['self_employed']),
            'ApplicantIncome': float(request.form['applicant_income']),
            'CoapplicantIncome': float(request.form['coapplicant_income']),
            'LoanAmount': float(request.form['loan_amount']),
            'Loan_Amount_Term': float(request.form['loan_term']),
            'Credit_History': int(request.form['credit_history']),
            'Property_Area': int(request.form['property_area']),
        }

        df = pd.DataFrame([user_input])

        # Apply scaling to numeric columns only
        num_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
        df[num_cols] = scaler.transform(df[num_cols])

        # Make prediction
        prediction = model.predict(df)[0]
        result = "Loan Approved ✅" if prediction == 1 else "Loan Not Approved ❌"
        return render_template('result.html', prediction=result)
    
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
