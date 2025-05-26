import joblib
import pandas as pd

# Load the model and preprocessing pipeline
model = joblib.load('model.pkl')

# Function to preprocess input data and make predictions
def predict_model(data):
    # Convert the input data into a DataFrame
    new_data = pd.DataFrame([data])
    
    # Define numerical and categorical columns
    numeric_features = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak']
    categorical_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
    
    # Ensure all required columns are present
    for feature in numeric_features + categorical_features:
        if feature not in new_data.columns:
            new_data[feature] = 0  # Add missing columns with default value 0
    
    # Ensure columns are in the correct order expected by the model
    new_data = new_data[numeric_features + categorical_features]
    
    # Use the loaded model to predict
    prediction = model.predict(new_data)
    return "Heart is safe" if prediction[0] == 0 else "Heart is not safe"

# Example usage
data = {
    'age': 45,
    'sex': 1,
    'cp': 0,
    'trestbps': 120,
    'chol': 240,
    'fbs': 1,
    'restecg': 0,
    'thalch': 150,
    'exang': 0,
    'oldpeak': 1.0,
    'slope': 0,
    'ca': 0,
    'thal': 0
}

result = predict_model(data)
print(result)
