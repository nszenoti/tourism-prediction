import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download
from custom_transformers import FeatureSelector, CategoricalMarker

@st.cache_resource
def load_model():
    """Load model from HuggingFace"""
    try:
        model_path = hf_hub_download(
            repo_id="nszfgfg/tourism-model",
            filename="selected_model.joblib"
        )
        return joblib.load(model_path)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def main():
    # Load the model
    model = load_model()

    if model is None:
        st.error("Failed to load model. Please check the model path and try again.")
        return

    # Title and Description
    st.title('Tourism Package Purchase Predictor')
    st.write('Enter customer information to predict likelihood of package purchase')

    # Personal Information
    st.header('Personal Details')
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input('Age', min_value=18, max_value=100)
        gender = st.selectbox('Gender', ['Male', 'Female'])
        marital_status = st.selectbox('Marital Status', ['Single', 'Married', 'Divorced'])
        own_car = st.selectbox('Owns Car?', ['Yes', 'No'])
    with col2:
        occupation = st.selectbox('Occupation', ['Salaried', 'Free Lancer', 'Business'])
        monthly_income = st.number_input('Monthly Income', min_value=0)
        designation = st.selectbox('Designation', ['Executive', 'Manager', 'Senior Manager', 'AVP', 'VP'])

    # Travel Preferences
    st.header('Travel Information')
    col1, col2 = st.columns(2)
    with col1:
        num_trips = st.number_input('Number of Past Trips', min_value=0)
        preferred_star = st.selectbox('Preferred Property Star Rating', [3, 4, 5])
        city_tier = st.selectbox('City Tier', [1, 2, 3])
    with col2:
        passport = st.selectbox('Has Passport?', ['Yes', 'No'])
        num_persons = st.number_input('Number of Persons Visiting', min_value=1)
        num_children = st.number_input('Number of Children Visiting', min_value=0)

    # Sales Information
    st.header('Sales Interaction')
    col1, col2 = st.columns(2)
    with col1:
        product_pitched = st.selectbox('Product Pitched', ['Basic', 'Standard', 'Deluxe', 'Super Deluxe', 'King'])
        pitch_duration = st.number_input('Duration of Pitch (minutes)', min_value=1)
        pitch_satisfaction = st.slider('Pitch Satisfaction Score', 1, 5)
    with col2:
        type_contact = st.selectbox('Type of Contact', ['Self Enquiry', 'Company Invited'])
        num_followups = st.number_input('Number of Followups', min_value=0)

    # Predict button
    if st.button('Predict Purchase Likelihood'):
        input_data = pd.DataFrame({
            'Age': [age],
            'Gender': [gender],
            'MaritalStatus': [marital_status],
            'Occupation': [occupation],
            'MonthlyIncome': [monthly_income],
            'Designation': [designation],
            'NumberOfTrips': [num_trips],
            'PreferredPropertyStar': [preferred_star],
            'NumberOfPersonVisiting': [num_persons],
            'NumberOfChildrenVisiting': [num_children],
            'Passport': [1 if passport == 'Yes' else 0],
            'CityTier': [city_tier],
            'TypeofContact': [type_contact],
            'DurationOfPitch': [pitch_duration],
            'NumberOfFollowups': [num_followups],
            'PitchSatisfactionScore': [pitch_satisfaction],
            'OwnCar': [1 if own_car == 'Yes' else 0],
            'ProductPitched': [product_pitched]
        })

        # Make prediction
        prediction = model.predict_proba(input_data)[0]

        # Show results
        st.subheader('Prediction Results')
        st.write(f'Likelihood of purchasing package: {prediction[1]:.2%}')

if __name__ == "__main__":
    main()
