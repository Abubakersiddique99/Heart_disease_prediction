import joblib

# List of feature names used to train the model
# Replace this list with the actual feature names from your training dataset
feature_names = [
    'age', 'sex', 'cp_typical angina', 'cp_non-anginal pain', 'cp_asymptomatic', 
    'trestbps', 'chol', 'fbs', 'restecg_1', 'restecg_2', 
    'thalach', 'exang', 'oldpeak', 'slope_1', 'slope_2', 
    'ca', 'thal_fixed defect', 'thal_reversable defect'
]

# Save the feature names to a pickle file
joblib.dump(feature_names, 'feature_names.pkl')
