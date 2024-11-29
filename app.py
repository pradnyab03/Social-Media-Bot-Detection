import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib


# Load the trained model and scaler
best_model = joblib.load('model/rf_model.pkl')  # Load your trained model
scaler = joblib.load('model/scaler.pkl')  # Load the saved scaler 
feature_names = np.load('model/feature_names.npy', allow_pickle=True)

# Function to preprocess text columns
def preprocess_text_columns(df):
    for column in ['lang', 'location']:
        df[column] = df[column].str.replace('"', '', regex=False).str.lower()
    df['verified'] = df['verified'].astype(int)
    df['default_profile'] = df['default_profile'].astype(int)
    return df

# Streamlit interface
st.title("Social Media Bot Detection")
st.write("Enter the details below to predict if a user is a bot or not:")

# User input fields
followers_count = st.number_input("Followers Count", min_value=0)
friends_count = st.number_input("Friends Count", min_value=0)
listed_count = st.number_input("Listed Count", min_value=0)
favourites_count = st.number_input("Favourites Count", min_value=0)
verified = st.selectbox("Verified", (False, True))
statuses_count = st.number_input("Statuses Count", min_value=0)
default_profile = st.selectbox("Default Profile", (False, True))
location_missing = st.selectbox("Location Missing (1 for Yes, 0 for No)", (1, 0))
lang = st.selectbox("Language", ('en', 'fr', 'es', 'de', 'it', 'pt', 'zh', 'other'))  # Add more options as needed
location = st.text_input("Location", "")

# Button to predict
if st.button("Predict"):
    # Prepare the input DataFrame
    new_input = pd.DataFrame({
        'followers_count': [followers_count],
        'friends_count': [friends_count],
        'listed_count': [listed_count],
        'favourites_count': [favourites_count],
        'verified': [verified],
        'statuses_count': [statuses_count],
        'default_profile': [default_profile],
        'location_missing': [location_missing],
        'lang': [lang],
        'location': [location]
    })

    # Preprocess the input
    new_input = preprocess_text_columns(new_input)
    new_input = pd.get_dummies(new_input, columns=['location', 'lang'], prefix=['location', 'lang'])
    
    # Remove 'bot' from feature names and reindex
    feature_names = feature_names[feature_names != 'bot']
    new_input = new_input.reindex(columns=feature_names, fill_value=0)

    # Scale the input
    new_input_scaled = scaler.transform(new_input)
    
    # Make prediction
    y_pred = best_model.predict(new_input_scaled)

    # Display the prediction result
    if y_pred[0] == 1:
        st.success("Prediction: This user is likely a bot.")
    else:
        st.success("Prediction: This user is likely not a bot.")
