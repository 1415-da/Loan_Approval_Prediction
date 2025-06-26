import os
from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__, template_folder="templates")

# Load the model and scaler
model = joblib.load(os.path.join(os.path.dirname(__file__), 'loan_status_predictor.pkl'))
scaler = joblib.load(os.path.join(os.path.dirname(__file__), 'vector.pkl'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect data from form
        user_input = [
            int(request.form['gender']),
            int(request.form['married']),
            int(request.form['dependents']),
            int(request.form['education']),
            int(request.form['self_employed']),
            float(request.form['applicant_income']),
            float(request.form['coapplicant_income']),
            float(request.form['loan_amount']),
            float(request.form['loan_term']),
            int(request.form['credit_history']),
            int(request.form['property_area'])
        ]
        # Indices of numeric columns for scaling
        num_indices = [5, 6, 7, 8]
        user_input_np = np.array(user_input).reshape(1, -1)
        # Scale only numeric columns
        user_input_np[0, num_indices] = scaler.transform(user_input_np[0, num_indices].reshape(1, -1))
        prediction = model.predict(user_input_np)[0]
        result = "Loan Approved ✅" if prediction == 1 else "Loan Not Approved ❌"
        return render_template('result.html', prediction=result)
    except Exception as e:
        return f"Error: {str(e)}"

def handler(environ, start_response):
    from werkzeug.middleware.dispatcher import DispatcherMiddleware
    from werkzeug.serving import run_simple
    application = DispatcherMiddleware(app)
    return application(environ, start_response) 
