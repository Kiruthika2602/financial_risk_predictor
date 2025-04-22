from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model and encoders
with open('model/risk_model.pkl', 'rb') as f:
    data = pickle.load(f)
    model = data['model']
    encoders = data['encoders']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    credit_rating = request.form['credit_rating']
    employment_type = request.form['employment_type']
    account_status = request.form['account_status']
    income_bracket = request.form['income_bracket']

    # Encode inputs
    input_data = [
        encoders['credit_rating'].transform([credit_rating])[0],
        encoders['employment_type'].transform([employment_type])[0],
        encoders['account_status'].transform([account_status])[0],
        encoders['income_bracket'].transform([income_bracket])[0]
    ]

    # Predict risk
    prediction = model.predict([input_data])[0]
    risk = encoders['risk'].inverse_transform([prediction])[0]

    return render_template('result.html', prediction=risk)

if __name__ == '__main__':
    app.run(debug=True)
