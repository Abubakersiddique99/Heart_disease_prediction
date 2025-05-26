import joblib
import pandas as pd
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Load the model
model = joblib.load('model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        data = request.form.to_dict()
        
        # Convert data into DataFrame
        new_data = pd.DataFrame([data])
        
        # Convert numeric fields
        numeric_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
        for feature in numeric_features:
            new_data[feature] = pd.to_numeric(new_data[feature])

        # Ensure the new data has the same columns as the training data
        categorical_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
        all_features = numeric_features + categorical_features
        
        for feature in all_features:
            if feature not in new_data.columns:
                new_data[feature] = 0  # Add missing columns with default value 0
        
        # Reorder the columns to match the training data
        new_data = new_data[all_features]
        
        # Make prediction
        prediction = model.predict(new_data)
        result = "Heart is safe" if prediction[0] == 0 else "Heart is not safe"
        
        return jsonify({'prediction': result})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
