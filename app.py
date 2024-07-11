import pickle
import streamlit as st
import numpy as np

# Load the trained model and scaler from a file
with open('svm_classifier.pkl', 'rb') as f:
    model_data = pickle.load(f)
    svm_classifier = model_data['model']  # Assuming your model was saved as 'model' in pickle
    ss_train_test = model_data['scaler']  # Assuming your scaler was saved as 'scaler' in pickle

# Define the features that the user can input
features = ['Umur', 'Jenis Kelamin', 'TB', 'BB']

# Create a form to collect user input
form = st.form("my_form")
with form:
    # Add input fields for each feature
    for feature in features:
        st.text_input(label=feature, key=feature)

    # Submit button
    submit_button = st.form_submit_button(label="Predict")

# If the form is submitted, predict the label
if submit_button:
    # Get the user input from the form
    user_input = {}
    for feature in features:
        user_input[feature] = st.session_state[feature]

    # Convert the user input to a numpy array
    user_input_array = np.array([list(user_input.values())])

    # Standardize the user input using the loaded scaler
    user_input_ss_scaled = ss_train_test.transform(user_input_array)

    # Predict the label
    prediction = svm_classifier.predict(user_input_ss_scaled)[0]

    # Display the prediction
    st.write("Predicted label:", prediction)
