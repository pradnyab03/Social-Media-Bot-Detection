import pickle
import pandas as pd

# Load the scaler
with open('model/scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Load the training features (X) to use for reindexing
X = pd.read_csv(r'C:\Users\Pradnya\OneDrive\OfficeMobile\Downloads\training_data_2_csv_UTF.csv')  # Load the training feature set

def preprocess_text_columns(input_data):
    # Add any necessary text preprocessing steps here (if needed)
    return input_data

def preprocess_input_data(input_data):
    # Preprocessing steps (cleaning, encoding, etc.)
    input_data = preprocess_text_columns(input_data)  # Your existing preprocessing function
    input_data = pd.get_dummies(input_data, columns=['location', 'lang'], prefix=['location', 'lang'])
    
    # Reindex to match training columns
    input_data = input_data.reindex(columns=X.columns, fill_value=0)  # Ensure input has same features as training set

    # Scale the input data
    input_data_scaled = scaler.transform(input_data)  # Use transform here
    return input_data_scaled
