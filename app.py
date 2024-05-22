import streamlit as st
import pandas as pd
import joblib
import pycountry
import shap
import matplotlib.pyplot as plt

# Title of the application
st.title('Hotel Booking Prediction')

# Load the pre-trained pipeline with error handling
pipeline_path = 'lgbm-pipeline.pkl'
try:
    pipeline = joblib.load(pipeline_path)
    st.success('Pipeline loaded successfully.')
except FileNotFoundError:
    st.error(f"Pipeline file not found: {pipeline_path}")
    st.stop()
except Exception as e:
    st.error(f"An error occurred while loading the pipeline: {e}")
    st.stop()

# Function to validate ISO country codes


def is_valid_country_code(code):
    return pycountry.countries.get(alpha_3=code) is not None


# Numerical Features
total_of_special_requests = st.number_input(
    'Total of Special Requests',
    min_value=0, max_value=5, value=0,
    help='Number of special requests made by the guest (e.g., extra bed, high floor, etc.)'
)
is_repeated_guest = st.selectbox(
    'Is Repeated Guest',
    options=[0, 1],
    help='Indicates whether the guest is a returning guest'
)
lead_time = st.number_input(
    'Lead Time (days)',
    min_value=0, value=0,
    help='Number of days between the booking date and the arrival date'
)
adults = st.number_input(
    'Number of Adults',
    min_value=1, max_value=10, value=1,
    help='Number of adults in the booking'
)
required_car_parking_spaces = st.number_input(
    'Required Car Parking Spaces',
    min_value=0, max_value=3, value=0,
    help='Number of car parking spaces required by the guest'
)
booking_changes = st.number_input(
    'Booking Changes',
    min_value=0, value=0,
    help='Number of changes made to the booking'
)
previous_cancellations = st.number_input(
    'Previous Cancellations',
    min_value=0, value=0,
    help='Number of previous bookings that were cancelled by the guest'
)
agent = st.number_input(
    'Agent ID (0 if none)',
    min_value=0, value=0,
    help='ID of the travel agency that made the booking (if any)'
)
company = st.number_input(
    'Company ID (0 if none)',
    min_value=0, value=0,
    help='ID of the company that made the booking (if any)'
)
arrival_date_year = st.number_input(
    'Arrival Date Year',
    min_value=2000, value=2024,
    help='Year of the arrival date'
)
stays_in_week_nights = st.number_input(
    'Stays in Week Nights',
    min_value=0, value=0,
    help='Number of week nights the guest will stay'
)
babies = st.number_input(
    'Number of Babies',
    min_value=0, value=0,
    help='Number of babies in the booking'
)
previous_bookings_not_canceled = st.number_input(
    'Previous Bookings Not Canceled',
    min_value=0, value=0,
    help='Number of previous bookings that were not canceled'
)
days_in_waiting_list = st.number_input(
    'Days in Waiting List',
    min_value=0, value=0,
    help='Number of days the booking was on the waiting list'
)
adr = st.number_input(
    'Average Daily Rate (ADR)',
    min_value=0.0, value=0.0,
    help='Average daily rate for the booking'
)

# Categorical Features
hotel = st.selectbox(
    'Hotel Type',
    options=['Resort Hotel', 'City Hotel'],
    help='Type of hotel'
)
country = st.text_input(
    'Country (ISO Code)',
    value='',
    help='Country of origin of the guest (ISO 3166-1 alpha-3 code)'
)
if country and not is_valid_country_code(country):
    st.error("Invalid ISO country code. Please enter a valid 3-letter ISO code.")

market_segment = st.selectbox(
    'Market Segment',
    options=['Direct', 'Corporate', 'Online TA', 'Offline TA/TO'],
    help='Market segment designation'
)
distribution_channel = st.selectbox(
    'Distribution Channel',
    options=['Direct', 'Corporate', 'TA/TO', 'GDS'],
    help='Distribution channel through which the booking was made'
)
deposit_type = st.selectbox(
    'Deposit Type',
    options=['No Deposit', 'Refundable', 'Non Refundable'],
    help='Type of deposit required for the booking'
)
customer_type = st.selectbox(
    'Customer Type',
    options=['Transient', 'Contract', 'Group', 'Transient-Party'],
    help='Type of booking (customer category)'
)
reservation_status = st.selectbox(
    'Reservation Status',
    options=['Canceled', 'Check-Out', 'No-Show'],
    help='Status of the reservation'
)
reservation_month = st.selectbox(
    'Reservation Month',
    options=[f'Month {i}' for i in range(1, 13)],
    help='Month of the reservation'
)
reservation_day = st.selectbox(
    'Reservation Day',
    options=[f'Day {i}' for i in range(1, 32)],
    help='Day of the reservation'
)
arrival_week = st.selectbox(
    'Arrival Week',
    options=[f'Week {i}' for i in range(1, 53)],
    help='Week of the year of the arrival date'
)
reserved_room_type = st.selectbox(
    'Reserved Room Type',
    options=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'L'],
    help='Code of the reserved room type'
)
arrival_date_month = st.selectbox(
    'Arrival Date Month',
    options=[f'Month {i}' for i in range(1, 13)],
    help='Month of the arrival date'
)
arrival_day = st.number_input(
    'Arrival Day',
    min_value=1, max_value=31, value=1,
    help='Day of the arrival date'
)
meal = st.selectbox(
    'Meal',
    options=['BB', 'HB', 'FB', 'SC', 'Undefined'],
    help='Type of meal booked'
)
assigned_room_type = st.selectbox(
    'Assigned Room Type',
    options=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'P'],
    help='Code for the assigned room type'
)

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
    'arrival_date_year': [arrival_date_year],
    'stays_in_week_nights': [stays_in_week_nights],
    'babies': [babies],
    'previous_bookings_not_canceled': [previous_bookings_not_canceled],
    'days_in_waiting_list': [days_in_waiting_list],
    'adr': [adr],
    'hotel': [hotel],
    'country': [country],
    'market_segment': [market_segment],
    'distribution_channel': [distribution_channel],
    'deposit_type': [deposit_type],
    'customer_type': [customer_type],
    'reservation_status': [reservation_status],
    'reservation_month': [reservation_month],
    'reservation_day': [reservation_day],
    'arrival_week': [arrival_week],
    'reserved_room_type': [reserved_room_type],
    'arrival_date_month': [arrival_date_month],
    'meal': [meal],
    'assigned_room_type': [assigned_room_type],
    'arrival_day': [arrival_day]
})

# Button to submit the form and make predictions
if st.button('Predict with Details'):
    if country and not is_valid_country_code(country):
        st.error("Please enter a valid 3-letter ISO country code.")
    else:
        try:
            # Transform the input data
            transformed_input_data = pipeline.named_steps['preprocessor'].transform(
                input_data)

            # Retrieve the feature names after transformation
            feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out(
            )

            # Convert the transformed data to a DataFrame with the correct feature names
            transformed_df = pd.DataFrame(
                transformed_input_data, columns=feature_names)

            # Make predictions
            prediction = pipeline.named_steps['model'].predict(transformed_df)
            prediction_proba = pipeline.named_steps['model'].predict_proba(
                transformed_df)

            # Display the prediction
            st.markdown(f'### Prediction: {prediction[0]}')
            st.markdown(f'### Prediction Probability:')
            st.write(prediction_proba[0])

            # Provide a description of the prediction
            if prediction[0] == 1:
                st.markdown(
                    '#### The model predicts that the booking will be **successful**.')
                st.markdown(
                    'This means that the guest is likely to show up and use the booking as planned.')
            else:
                st.markdown(
                    '#### The model predicts that the booking might **not be successful**.')
                st.markdown(
                    'This could mean the booking might be canceled or the guest might not show up.')

            # Provide additional context or suggestions based on the prediction
            if prediction[0] == 0 and prediction_proba[0][0] > 0.7:
                st.markdown(
                    '**There is a high probability that the booking will be unsuccessful. Consider reaching out to the guest for confirmation.**')
            elif prediction[0] == 1 and prediction_proba[0][1] > 0.7:
                st.markdown('**There is a high probability that the booking will be successful. Ensure that the special requests and other preferences are noted for better customer satisfaction.**')

            # Explain the prediction using SHAP
            explainer = shap.TreeExplainer(
                pipeline.named_steps['model'], feature_perturbation="tree_path_dependent")
<< << << < HEAD
            shap_values = explainer.shap_values(transformed_df)
== == == =
            shap_values = explainer.shap_values(input_data)
>>>>>> > parent of 8c92ce8(test the fix)

            # Plot SHAP values with correct feature names
            st.markdown("### Feature Importance based on SHAP values")
            fig, ax = plt.subplots()
<< << << < HEAD
            shap.summary_plot(shap_values, transformed_df,
                              feature_names=feature_names, show=False)
== == == =
            shap.summary_plot(shap_values, input_data, show=False)
>>>>>> > parent of 8c92ce8(test the fix)
            st.pyplot(fig)

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
