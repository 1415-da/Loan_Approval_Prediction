# 🚀 Loan Approval Prediction Flask App

**Loan Approval Prediction** is a web application built using Flask to predict whether a loan will be approved based on user input. The application uses a pre-trained machine learning model and provides predictions based on applicant details such as income, credit history, and other factors.

## ✨ Features

- **Fully Responsive** – The web interface is mobile-friendly and adapts to various screen sizes.
- **Loan Prediction** – Predicts whether a loan will be approved or denied based on the user's input.
- **Machine Learning Model** – Trained model using real-world data to predict loan approval status.
- **Clean UI** – Minimal, easy-to-use interface for users to input their data.
- **Secure** – Data is handled securely with form validation and proper error handling.

## 🚀 Technologies Used

- **Backend**: Python, Flask
- **Machine Learning**: scikit-learn, joblib
- **Frontend**: HTML, CSS (Jinja templating)
- **Data Handling**: pandas, numpy

## 💻 Model Training

To train the model, follow these steps:

1. Place the loan dataset (for training) in the same directory as `model_training.py`.

2. Run the training script:

    ```bash
    python model_training.py
    ```

3. The script will:

   - Train various classifiers (Logistic Regression, Random Forest, etc.).
   - Evaluate the models and choose the best-performing one.
   - Save the trained model as `loan_status_predictor.pkl` and the scaler as `vector.pkl` in the `model/` directory.

4. After training, you can use the trained model to make predictions with the Flask app.

## 📂 Files

- `app.py`: The main Flask application that handles user input and model prediction.
- `templates/index.html`: The form where users input their loan details.
- `templates/result.html`: The result page that shows whether the loan is approved or not.
- `model/loan_status_predictor.pkl`: The trained machine learning model (saved using `joblib`).
- `model/vector.pkl`: The scaler used to scale the numerical input features.
- `model_training.py`: Script to train and save the machine learning model.
