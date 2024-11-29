import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib  # To load the model and scaler

# Load the trained model and scaler
best_model = joblib.load('model/rf_model.pkl')  # Load your trained model
scaler = joblib.load('model/scaler.pkl')  # Load the saved scaler

# Load the saved feature names used during model training
feature_names = np.load('model/feature_names.npy', allow_pickle=True)

# New input data provided by the user
new_input = pd.DataFrame({
    'followers_count': [1086],
    'friends_count': [0], 
    'listed_count': [14], 
    'favourites_count': [0],
    'verified': [False], 
    'statuses_count': [713], 
    'default_profile': [True], 
    'location_missing': [1],
    'lang': ['en'], 
    'location': ['']
})

# Function to preprocess text columns
def preprocess_text_columns(df):
    for column in ['lang', 'location']:
        # Remove quotes and convert to lowercase
        df[column] = df[column].str.replace('"', '', regex=False).str.lower()
    # Convert booleans to integers
    df['verified'] = df['verified'].astype(int)
    df['default_profile'] = df['default_profile'].astype(int)
    return df

# Preprocess the new input
new_input = preprocess_text_columns(new_input)

# One-hot encode the categorical columns 'location' and 'lang'
new_input = pd.get_dummies(new_input, columns=['location', 'lang'], prefix=['location', 'lang'])

# Reindex the new input to match the feature names used during training
# Drop 'bot' from feature_names to avoid mismatch
feature_names = feature_names[feature_names != 'bot']  # Remove 'bot' if it exists
new_input = new_input.reindex(columns=feature_names, fill_value=0)

# Scale the input using the same scaler that was used during training
new_input_scaled = scaler.transform(new_input)

# Convert scaled data back to a DataFrame for easy interpretation
new_input_scaled_df = pd.DataFrame(new_input_scaled, columns=new_input.columns)

# Make a prediction using the model
y_pred = best_model.predict(new_input_scaled_df)
print("Prediction:", y_pred)
