import streamlit as st
import pandas as pd
import joblib

# Load the pre-trained pipeline
pipeline = joblib.load('lgbm-pipeline.pkl')

# Title of the application
st.title('Hotel Booking Prediction')

# Numerical Features
total_of_special_requests = st.number_input(
    'Total of Special Requests', min_value=0, max_value=5, value=0)
is_repeated_guest = st.selectbox('Is Repeated Guest', options=[0, 1])
lead_time = st.number_input('Lead Time (days)', min_value=0, value=0)
adults = st.number_input(
    'Number of Adults', min_value=1, max_value=10, value=1)
required_car_parking_spaces = st.number_input(
    'Required Car Parking Spaces', min_value=0, max_value=3, value=0)
booking_changes = st.number_input('Booking Changes', min_value=0, value=0)
previous_cancellations = st.number_input(
    'Previous Cancellations', min_value=0, value=0)
agent = st.number_input('Agent ID (0 if none)', min_value=0, value=0)
company = st.number_input('Company ID (0 if none)', min_value=0, value=0)

# Categorical Features
hotel = st.selectbox('Hotel Type', options=['Resort Hotel', 'City Hotel'])
country = st.text_input('Country (ISO Code)', value='')
market_segment = st.selectbox('Market Segment', options=[
                              'Direct', 'Corporate', 'Online TA', 'Offline TA/TO'])
distribution_channel = st.selectbox('Distribution Channel', options=[
                                    'Direct', 'Corporate', 'TA/TO', 'GDS'])
deposit_type = st.selectbox('Deposit Type', options=[
                            'No Deposit', 'Refundable', 'Non Refundable'])
customer_type = st.selectbox('Customer Type', options=[
                             'Transient', 'Contract', 'Group', 'Transient-Party'])
reservation_status = st.selectbox('Reservation Status', options=[
                                  'Canceled', 'Check-Out', 'No-Show'])
reservation_month = st.selectbox('Reservation Month', options=[
                                 f'Month {i}' for i in range(1, 13)])
reservation_day = st.selectbox('Reservation Day', options=[
                               f'Day {i}' for i in range(1, 32)])

# Prepare the input data as a DataFrame
input_data = pd.DataFrame({
    'total_of_special_requests': [total_of_special_requests],
    'is_repeated_guest': [is_repeated_guest],
    'lead_time': [lead_time],
    'adults': [adults],
    'required_car_parking_spaces': [required_car_parking_spaces],
    'booking_changes': [booking_changes],
    'previous_cancellations': [previous_cancellations],
    'agent': [agent],
    'company': [company],
    'hotel': [hotel],
    'country': [country],
    'market_segment': [market_segment],
    'distribution_channel': [distribution_channel],
    'deposit_type': [deposit_type],
    'customer_type': [customer_type],
    'reservation_status': [reservation_status],
    'reservation_month': [reservation_month],
    'reservation_day': [reservation_day]
})

# Button to submit the form and make predictions
if st.button('Predict'):
    # Make predictions
    prediction = pipeline.predict(input_data)

    # Display the prediction
    st.write('Prediction:', prediction[0])
