
from flask import Flask, render_template, request
import pickle
import numpy as np

# Load trained model
filename = 'best_heart_model.pkl'
model = pickle.load(open(filename, 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('main.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Convert inputs and validate
        age = int(request.form['age'])
        sex = int(request.form['sex'])
        cp = int(request.form['cp'])
        trestbps = int(request.form['trestbps'])
        chol = int(request.form['chol'])
        fbs = int(request.form['fbs'])
        restecg = int(request.form['restecg'])
        thalach = int(request.form['thalach'])
        exang = int(request.form['exang'])
        oldpeak = float(request.form['oldpeak'])
        slope = int(request.form['slope'])
        ca = int(request.form['ca'])
        thal = int(request.form['thal'])
         # Combine into array
        data = np.array([[age, sex, cp, trestbps, chol, fbs,
                          restecg, thalach, exang, oldpeak, slope, ca, thal]])

        # 🔍 Debug print
        print("Raw input data:", data)

        # Predict
        my_prediction = model.predict(data)

        # 🔍 Debug print
        print("Prediction result:", my_prediction)

        # Example validations
        if not (0 <= ca <= 4):
            raise ValueError("Number of major vessels (ca) must be between 0 and 4.")
        if not (94 <= trestbps <= 200):
            raise ValueError("Resting blood pressure must be between 94 and 200 mmHg.")
        if not (126 <= chol <= 564):
            raise ValueError("Cholesterol must be between 126 and 564 mg/dl.")
        if not (71 <= thalach <= 202):
            raise ValueError("Maximum heart rate must be between 71 and 202 bpm.")
        if not (0 <= oldpeak <= 6.2):
            raise ValueError("ST depression (oldpeak) must be between 0.0 and 6.2.")

        # Make prediction
        data = np.array([[age, sex, cp, trestbps, chol, fbs,
                          restecg, thalach, exang, oldpeak, slope, ca, thal]])
        my_prediction = model.predict(data)

        return render_template('result.html', prediction=my_prediction[0])

    except ValueError as ve:
        return render_template('main.html', error=str(ve))
    except Exception:
        return render_template('main.html', error="Invalid input. Please enter valid values.")

if __name__ == '__main__':
    app.run(debug=True)

