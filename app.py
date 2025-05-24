import numpy as np
import joblib
import streamlit as st
import pandas as pd
import datetime
import altair as alt

# --- Streamlit Page Configuration (Theming and Layout) ---
st.set_page_config(
    page_title="Diabetes Risk Prediction & Prevention Hub",
    page_icon="🩸", # Blood drop icon
    layout="wide", # Use wide layout for more space
    initial_sidebar_state="expanded"
)

# Custom CSS for styling (white and blue theme)
st.markdown("""
    <style>
    .reportview-container {
        background: #FFFFFF; /* White background */
    }
    .sidebar .sidebar-content {
        background: #F0F8FF; /* Light blue for sidebar */
    }
    h1, h2, h3, h4, h5, h6 {
        color: #0056b3; /* Darker blue for headers */
    }
    .stSelectbox, .stNumberInput, .stTextInput, .stTextArea {
        border-color: #ADD8E6; /* Light blue border for inputs */
        border-radius: 5px;
    }
    .stButton>button {
        background-color: #007bff; /* Primary blue for buttons */
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 20px;
        font-size: 16px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #0056b3; /* Darker blue on hover */
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    .stAlert {
        border-radius: 8px;
    }
    .stSuccess {
        background-color: #e6ffed; /* Lighter green for success */
        color: #1a522d;
        border-left: 5px solid #28a745;
    }
    .stWarning {
        background-color: #fff3e6; /* Lighter orange for warning */
        color: #664100;
        border-left: 5px solid #ffc107;
    }
    .stError {
        background-color: #ffe6e6; /* Lighter red for error */
        color: #7d2626;
        border-left: 5px solid #dc3545;
    }
    .stInfo {
        background-color: #e6f7ff; /* Lighter blue for info */
        color: #004085;
        border-left: 5px solid #17a2b8;
    }
    .markdown-text-container {
        font-family: 'Arial', sans-serif;
        line-height: 1.6;
    }
    </style>
    """, unsafe_allow_html=True)


# --- Load the Model ---
try:
    model_path = 'trained_model.joblib'
    loaded_model = joblib.load(model_path)
except FileNotFoundError:
    st.error("🚨 Error: Trained model file not found. Please ensure 'trained_model.joblib' is in the correct directory. Cannot perform risk assessment without the model.")
    loaded_model = None

# --- Default Values ---
DEFAULT_AGE = 30
DEFAULT_WEIGHT = 70
DEFAULT_HEIGHT = 1.70
DEFAULT_PREGNANCIES = 0
DEFAULT_GLUCOSE = 100
DEFAULT_BLOODPRESSURE = 80
DEFAULT_SKINTHICKNESS = 20
DEFAULT_INSULIN = 50
DEFAULT_DIABETESPEDIGREEFUNCTION = 0.5

# --- Prediction Function with Probability Output ---
def diabetes_prediction_proba(input_data, model):
    if model is None:
        return None
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    prediction_proba = model.predict_proba(input_data_reshaped)
    return prediction_proba[0][1] if len(prediction_proba[0]) > 1 else prediction_proba[0][0]

# --- Utility Functions ---
def calculate_bmi(weight, height):
    if height > 0:
        return weight / (height ** 2)
    return 0.0

def get_risk_level(probability):
    if probability is None:
        return "N/A"
    if probability < 0.20:
        return "Low"
    elif probability < 0.50:
        return "Moderate"
    else:
        return "Elevated"

def display_risk_interpretation(risk_level, probability):
    st.markdown("---") # Separator
    st.subheader("📊 Your Risk Assessment Results")
    if risk_level == "Low":
        st.success(f"**Low Risk!** Your estimated risk of developing diabetes is **{probability * 100:.2f}%**. 🎉 Continue to maintain your healthy lifestyle choices. Regular check-ups are always recommended.")
    elif risk_level == "Moderate":
        st.warning(f"**Moderate Risk!** Your estimated risk of developing diabetes is **{probability * 100:.2f}%**. Consider making proactive lifestyle adjustments and scheduling regular health check-ups to monitor your status.")
    elif risk_level == "Elevated":
        st.error(f"**Elevated Risk!** Your estimated risk of developing diabetes is **{probability * 100:.2f}%**. It is **strongly recommended** to consult with a healthcare professional as soon as possible for further evaluation, guidance, and personalized management strategies.")
    else:
        st.info("Risk assessment unavailable. Please ensure all input fields are correctly filled and the model is loaded.")
    st.markdown("---")

def load_assessment_history(user_id):
    # This function is a placeholder; in a real app, this would load from a database
    return st.session_state.get('assessments', {}).get(user_id, [])

# --- Risk Assessment Page ---
def risk_assessment_page():
    st.header("Assess Your Diabetes Risk Now")
    st.markdown("Fill in your health details below to get an estimated risk assessment. Don't worry if you don't have all the exact numbers; you can use the default values provided or leave them blank.")

    # Initialize session state for inputs if they don't exist
    # Using specific keys for each input ensures uniqueness
    input_keys = {
        'age': 'input_age', 'pregnancies': 'input_pregnancies', 'glucose': 'input_glucose',
        'blood_pressure': 'input_blood_pressure', 'skin_thickness': 'input_skin_thickness',
        'insulin': 'input_insulin', 'weight': 'input_weight', 'height': 'input_height',
        'diabetes_pedigree_function': 'input_diabetes_pedigree_function'
    }
    for key in input_keys.values():
        if key not in st.session_state:
            st.session_state[key] = None

    # Create tabs for Basic and Advanced Information
    tab1, tab2, tab3 = st.tabs(["🧍 Basic Information", "🩺 Clinical Parameters", "✨ Quick Tips"])

    with tab1:
        st.subheader("Your Personal Details")
        st.markdown("Provide fundamental information for a preliminary assessment.")
        col1_basic, col2_basic = st.columns(2)
        
        # Using a more descriptive label and help text
        age_input = col1_basic.number_input(
            'Age (Years)', min_value=0, max_value=120,
            value=st.session_state[input_keys['age']],
            help="Your current age in years. (e.g., 30)",
            key='age_input_key'
        )
        pregnancies_input = col1_basic.number_input(
            'Number of Pregnancies (Female Only)', min_value=0, max_value=20,
            value=st.session_state[input_keys['pregnancies']],
            help="For females, the number of times pregnant. (e.g., 0)",
            key='pregnancies_input_key'
        )

        weight_input = col2_basic.number_input(
            'Weight (kg)', min_value=20, max_value=300,
            value=st.session_state[input_keys['weight']],
            help="Your weight in kilograms. (e.g., 70)",
            key='weight_input_key'
        )
        height_input = col2_basic.number_input(
            'Height (meters)', min_value=0.5, max_value=3.0, step=0.01,
            value=st.session_state[input_keys['height']],
            help="Your height in meters. (e.g., 1.70 for 170cm)",
            key='height_input_key'
        )

        # Update session state with current input values
        st.session_state[input_keys['age']] = age_input
        st.session_state[input_keys['pregnancies']] = pregnancies_input
        st.session_state[input_keys['weight']] = weight_input
        st.session_state[input_keys['height']] = height_input

        # BMI calculation displayed live
        bmi = calculate_bmi(
            st.session_state[input_keys['weight']] if st.session_state[input_keys['weight']] is not None else DEFAULT_WEIGHT,
            st.session_state[input_keys['height']] if st.session_state[input_keys['height']] is not None else DEFAULT_HEIGHT
        )
        st.metric("Body Mass Index (BMI)", f"{bmi:.2f}", help="Calculated as weight (kg) / height (m)^2. A BMI between 18.5 and 24.9 is considered healthy.")
        st.session_state['BMI'] = bmi # Store BMI in session state for potential use later

    with tab2:
        st.subheader("Your Clinical Parameters")
        st.markdown("Input medical metrics if available. If unsure, leave blank to use typical default values.")
        col1_adv, col2_adv = st.columns(2)
        
        glucose_input = col1_adv.number_input(
            'Glucose Level (mg/dL)', min_value=0, max_value=300,
            value=st.session_state[input_keys['glucose']],
            help="Plasma glucose concentration (e.g., 100 mg/dL).",
            key='glucose_input_key'
        )
        blood_pressure_input = col1_adv.number_input(
            'Blood Pressure (mmHg)', min_value=0, max_value=200,
            value=st.session_state[input_keys['blood_pressure']],
            help="Diastolic blood pressure (e.g., 80 mmHg).",
            key='blood_pressure_input_key'
        )
        skin_thickness_input = col2_adv.number_input(
            'Skin Thickness (mm)', min_value=0, max_value=100,
            value=st.session_state[input_keys['skin_thickness']],
            help="Triceps skin fold thickness (e.g., 20 mm).",
            key='skin_thickness_input_key'
        )
        insulin_input = col2_adv.number_input(
            'Insulin Level (mu U/mL)', min_value=0, max_value=850,
            value=st.session_state[input_keys['insulin']],
            help="2-Hour serum insulin (e.g., 50 mu U/mL).",
            key='insulin_input_key'
        )
        diabetes_pedigree_function_input = col1_adv.number_input(
            'Diabetes Pedigree Function', min_value=0.0, max_value=3.0, step=0.001,
            value=st.session_state[input_keys['diabetes_pedigree_function']],
            help="A function that scores the likelihood of diabetes based on family history. (e.g., 0.5)",
            key='dpf_input_key'
        )

        # Update session state with current input values
        st.session_state[input_keys['glucose']] = glucose_input
        st.session_state[input_keys['blood_pressure']] = blood_pressure_input
        st.session_state[input_keys['skin_thickness']] = skin_thickness_input
        st.session_state[input_keys['insulin']] = insulin_input
        st.session_state[input_keys['diabetes_pedigree_function']] = diabetes_pedigree_function_input
    
    with tab3:
        st.subheader("Quick Tips for Data Input")
        st.info("""
            * **Don't know a value?** Leave the field blank and we'll use a common default.
            * **Units are important!** Ensure your weight is in kilograms (kg) and height in meters (m).
            * **Family history matters:** The Diabetes Pedigree Function quantifies genetic influence.
            * **When in doubt, consult a professional:** This tool is for informational purposes; always verify with a doctor.
        """)

    st.markdown("---") # Separator between inputs and button

    # Centered button with clear call to action
    col_btn1, col_btn2, col_btn3 = st.columns([1,2,1])
    with col_btn2:
        if st.button("✨ Get My Risk Assessment", use_container_width=True, type="primary):
            if loaded_model:
                # Use actual input values, or default if None (left blank by user)
                final_age = st.session_state[input_keys['age']] if st.session_state[input_keys['age']] is not None else DEFAULT_AGE
                final_pregnancies = st.session_state[input_keys['pregnancies']] if st.session_state[input_keys['pregnancies']] is not None else DEFAULT_PREGNANCIES
                final_glucose = st.session_state[input_keys['glucose']] if st.session_state[input_keys['glucose']] is not None else DEFAULT_GLUCOSE
                final_blood_pressure = st.session_state[input_keys['blood_pressure']] if st.session_state[input_keys['blood_pressure']] is not None else DEFAULT_BLOODPRESSURE
                final_skin_thickness = st.session_state[input_keys['skin_thickness']] if st.session_state[input_keys['skin_thickness']] is not None else DEFAULT_SKINTHICKNESS
                final_insulin = st.session_state[input_keys['insulin']] if st.session_state[input_keys['insulin']] is not None else DEFAULT_INSULIN
                final_weight = st.session_state[input_keys['weight']] if st.session_state[input_keys['weight']] is not None else DEFAULT_WEIGHT
                final_height = st.session_state[input_keys['height']] if st.session_state[input_keys['height']] is not None else DEFAULT_HEIGHT
                final_diabetes_pedigree_function = st.session_state[input_keys['diabetes_pedigree_function']] if st.session_state[input_keys['diabetes_pedigree_function']] is not None else DEFAULT_DIABETESPEDIGREEFUNCTION

                # Recalculate BMI with potentially defaulted weight/height for the prediction input
                final_bmi = calculate_bmi(final_weight, final_height)

                input_data = [final_pregnancies, final_glucose, final_blood_pressure, final_skin_thickness, final_insulin, final_bmi, final_diabetes_pedigree_function, final_age]
                risk_probability = diabetes_prediction_proba(input_data, loaded_model)
                risk_level = get_risk_level(risk_probability)

                # Display results immediately below the button
                display_risk_interpretation(risk_level, risk_probability)
            else:
                st.error("Model not loaded. Cannot assess risk. Please contact support if this issue persists.")

def assessment_history_page():
    st.header("📈 Your Diabetes Risk Trend")
    st.markdown("Track how your estimated diabetes risk and key health metrics have changed over time. This data is simulated for demonstration.")

    # Fabricated data for user 'minh' - adjusted for weekly view
    current_year = 2024
    minh_history_data = [
        {'timestamp': datetime.datetime(current_year, 6, 1), 'age': 34, 'pregnancies': 0, 'glucose': 115, 'blood_pressure': 82, 'skin_thickness': 26, 'insulin': 60, 'bmi': 25.0, 'diabetes_pedigree_function': 0.58, 'risk_probability': 0.35, 'risk_level': 'Moderate'},
        {'timestamp': datetime.datetime(current_year, 6, 8), 'age': 34, 'pregnancies': 0, 'glucose': 112, 'blood_pressure': 81, 'skin_thickness': 25, 'insulin': 58, 'bmi': 24.9, 'diabetes_pedigree_function': 0.57, 'risk_probability': 0.34, 'risk_level': 'Moderate'},
        {'timestamp': datetime.datetime(current_year, 6, 15), 'age': 34, 'pregnancies': 0, 'glucose': 109, 'blood_pressure': 80, 'skin_thickness': 24, 'insulin': 56, 'bmi': 24.8, 'diabetes_pedigree_function': 0.56, 'risk_probability': 0.33, 'risk_level': 'Moderate'},
        {'timestamp': datetime.datetime(current_year, 6, 22), 'age': 34, 'pregnancies': 0, 'glucose': 106, 'blood_pressure': 79, 'skin_thickness': 23, 'insulin': 54, 'bmi': 24.7, 'diabetes_pedigree_function': 0.55, 'risk_probability': 0.32, 'risk_level': 'Moderate'},
        {'timestamp': datetime.datetime(current_year, 7, 1), 'age': 34, 'pregnancies': 0, 'glucose': 108, 'blood_pressure': 80, 'skin_thickness': 24, 'insulin': 55, 'bmi': 24.8, 'diabetes_pedigree_function': 0.55, 'risk_probability': 0.32, 'risk_level': 'Moderate'},
        {'timestamp': datetime.datetime(current_year, 7, 8), 'age': 34, 'pregnancies': 0, 'glucose': 105, 'blood_pressure': 79, 'skin_thickness': 23, 'insulin': 53, 'bmi': 24.7, 'diabetes_pedigree_function': 0.54, 'risk_probability': 0.31, 'risk_level': 'Moderate'},
        {'timestamp': datetime.datetime(current_year, 7, 15), 'age': 34, 'pregnancies': 0, 'glucose': 102, 'blood_pressure': 78, 'skin_thickness': 22, 'insulin': 51, 'bmi': 24.6, 'diabetes_pedigree_function': 0.53, 'risk_probability': 0.30, 'risk_level': 'Moderate'},
        {'timestamp': datetime.datetime(current_year, 7, 22), 'age': 34, 'pregnancies': 0, 'glucose': 99, 'blood_pressure': 77, 'skin_thickness': 21, 'insulin': 49, 'bmi': 24.5, 'diabetes_pedigree_function': 0.52, 'risk_probability': 0.29, 'risk_level': 'Moderate'},
        {'timestamp': datetime.datetime(current_year, 8, 1), 'age': 34, 'pregnancies': 0, 'glucose': 102, 'blood_pressure': 78, 'skin_thickness': 22, 'insulin': 50, 'bmi': 24.5, 'diabetes_pedigree_function': 0.50, 'risk_probability': 0.28, 'risk_level': 'Moderate'},
        {'timestamp': datetime.datetime(current_year, 8, 8), 'age': 34, 'pregnancies': 0, 'glucose': 100, 'blood_pressure': 77, 'skin_thickness': 21, 'insulin': 48, 'bmi': 24.4, 'diabetes_pedigree_function': 0.49, 'risk_probability': 0.27, 'risk_level': 'Moderate'},
        {'timestamp': datetime.datetime(current_year, 8, 15), 'age': 34, 'pregnancies': 0, 'glucose': 98, 'blood_pressure': 76, 'skin_thickness': 20, 'insulin': 46, 'bmi': 24.3, 'diabetes_pedigree_function': 0.48, 'risk_probability': 0.26, 'risk_level': 'Low'},
        {'timestamp': datetime.datetime(current_year, 8, 22), 'age': 34, 'pregnancies': 0, 'glucose': 96, 'blood_pressure': 75, 'skin_thickness': 19, 'insulin': 44, 'bmi': 24.2, 'diabetes_pedigree_function': 0.47, 'risk_probability': 0.25, 'risk_level': 'Low'},
        {'timestamp': datetime.datetime(current_year, 9, 1), 'age': 35, 'pregnancies': 0, 'glucose': 98, 'blood_pressure': 76, 'skin_thickness': 20, 'insulin': 45, 'bmi': 24.2, 'diabetes_pedigree_function': 0.48, 'risk_probability': 0.25, 'risk_level': 'Low'},
        {'timestamp': datetime.datetime(current_year, 9, 8), 'age': 35, 'pregnancies': 0, 'glucose': 97, 'blood_pressure': 77, 'skin_thickness': 21, 'insulin': 43, 'bmi': 24.3, 'diabetes_pedigree_function': 0.47, 'risk_probability': 0.24, 'risk_level': 'Low'},
        {'timestamp': datetime.datetime(current_year, 9, 15), 'age': 35, 'pregnancies': 0, 'glucose': 96, 'blood_pressure': 78, 'skin_thickness': 22, 'insulin': 41, 'bmi': 24.4, 'diabetes_pedigree_function': 0.46, 'risk_probability': 0.23, 'risk_level': 'Low'},
        {'timestamp': datetime.datetime(current_year, 9, 22), 'age': 35, 'pregnancies': 0, 'glucose': 95, 'blood_pressure': 79, 'skin_thickness': 23, 'insulin': 39, 'bmi': 24.5, 'diabetes_pedigree_function': 0.45, 'risk_probability': 0.15, 'risk_level': 'Low'},
        {'timestamp': datetime.datetime(current_year, 10, 1), 'age': 35, 'pregnancies': 0, 'glucose': 95, 'blood_pressure': 80, 'skin_thickness': 22, 'insulin': 40, 'bmi': 24.5, 'diabetes_pedigree_function': 0.45, 'risk_probability': 0.15, 'risk_level': 'Low'},
        {'timestamp': datetime.datetime(current_year, 10, 8), 'age': 35, 'pregnancies': 0, 'glucose': 98, 'blood_pressure': 81, 'skin_thickness': 23, 'insulin': 43, 'bmi': 24.6, 'diabetes_pedigree_function': 0.46, 'risk_probability': 0.18, 'risk_level': 'Low'},
        {'timestamp': datetime.datetime(current_year, 10, 15), 'age': 35, 'pregnancies': 0, 'glucose': 101, 'blood_pressure': 82, 'skin_thickness': 24, 'insulin': 46, 'bmi': 24.7, 'diabetes_pedigree_function': 0.47, 'risk_probability': 0.21, 'risk_level': 'Moderate'},
        {'timestamp': datetime.datetime(current_year, 10, 22), 'age': 35, 'pregnancies': 0, 'glucose': 104, 'blood_pressure': 83, 'skin_thickness': 25, 'insulin': 49, 'bmi': 24.8, 'diabetes_pedigree_function': 0.48, 'risk_probability': 0.24, 'risk_level': 'Low'},
        {'timestamp': datetime.datetime(current_year, 11, 1), 'age': 35, 'pregnancies': 0, 'glucose': 105, 'blood_pressure': 83, 'skin_thickness': 25, 'insulin': 50, 'bmi': 24.7, 'diabetes_pedigree_function': 0.51, 'risk_probability': 0.29, 'risk_level': 'Moderate'},
        {'timestamp': datetime.datetime(current_year, 11, 8), 'age': 35, 'pregnancies': 0, 'glucose': 108, 'blood_pressure': 84, 'skin_thickness': 26, 'insulin': 53, 'bmi': 24.8, 'diabetes_pedigree_function': 0.52, 'risk_probability': 0.31, 'risk_level': 'Moderate'},
        {'timestamp': datetime.datetime(current_year, 11, 15), 'age': 35, 'pregnancies': 0, 'glucose': 111, 'blood_pressure': 85, 'skin_thickness': 27, 'insulin': 56, 'bmi': 24.9, 'diabetes_pedigree_function': 0.53, 'risk_probability': 0.33, 'risk_level': 'Moderate'},
        {'timestamp': datetime.datetime(current_year, 11, 22), 'age': 35, 'pregnancies': 0, 'glucose': 114, 'blood_pressure': 86, 'skin_thickness': 28, 'insulin': 59, 'bmi': 25.0, 'diabetes_pedigree_function': 0.54, 'risk_probability': 0.35, 'risk_level': 'Moderate'},
        {'timestamp': datetime.datetime(current_year, 12, 1), 'age': 35, 'pregnancies': 0, 'glucose': 112, 'blood_pressure': 86, 'skin_thickness': 27, 'insulin': 58, 'bmi': 24.9, 'diabetes_pedigree_function': 0.54, 'risk_probability': 0.33, 'risk_level': 'Moderate'},
        {'timestamp': datetime.datetime(current_year + 1, 1, 1), 'age': 35, 'pregnancies': 0, 'glucose': 118, 'blood_pressure': 88, 'skin_thickness': 29, 'insulin': 65, 'bmi': 25.1, 'diabetes_pedigree_function': 0.57, 'risk_probability': 0.38, 'risk_level': 'Moderate'},
        {'timestamp': datetime.datetime(current_year + 1, 2, 1), 'age': 36, 'pregnancies': 0, 'glucose': 110, 'blood_pressure': 85, 'skin_thickness': 26, 'insulin': 60, 'bmi': 25.3, 'diabetes_pedigree_function': 0.59, 'risk_probability': 0.40, 'risk_level': 'Moderate'},
        {'timestamp': datetime.datetime(current_year + 1, 3, 1), 'age': 36, 'pregnancies': 0, 'glucose': 105, 'blood_pressure': 83, 'skin_thickness': 24, 'insulin': 55, 'bmi': 25.0, 'diabetes_pedigree_function': 0.56, 'risk_probability': 0.37, 'risk_level': 'Moderate'},
        {'timestamp': datetime.datetime(current_year + 1, 4, 1), 'age': 36, 'pregnancies': 0, 'glucose': 100, 'blood_pressure': 81, 'skin_thickness': 22, 'insulin': 50, 'bmi': 24.8, 'diabetes_pedigree_function': 0.53, 'risk_probability': 0.30, 'risk_level': 'Moderate'},
        {'timestamp': datetime.datetime(current_year + 1, 5, 1), 'age': 36, 'pregnancies': 0, 'glucose': 96, 'blood_pressure': 79, 'skin_thickness': 20, 'insulin': 45, 'bmi': 24.6, 'diabetes_pedigree_function': 0.50, 'risk_probability': 0.26, 'risk_level': 'Low'},
    ]
    df_minh = pd.DataFrame(minh_history_data)
    df_minh['timestamp'] = pd.to_datetime(df_minh['timestamp'])
    df_minh = df_minh.sort_values(by='timestamp')

    st.subheader("Filter Your Trend Data:")
    col_view_by, col_metric = st.columns(2)

    with col_view_by:
        view_by = st.radio("View Data By:", ["Month", "Week"], horizontal=True, key="view_by_radio")

    if view_by == "Month":
        df_minh['time_period'] = df_minh['timestamp'].dt.to_period('M').astype(str)
        time_title = 'Month'
    else:
        # Format week to show year, e.g., "W22-2024"
        df_minh['time_period'] = 'W' + df_minh['timestamp'].dt.isocalendar().week.astype(str) + '-' + df_minh['timestamp'].dt.year.astype(str)
        time_title = 'Week'

    metrics_to_plot = {
        "Risk Probability": "risk_probability",
        "BMI": "bmi",
        "Glucose Level": "glucose",
        "Blood Pressure": "blood_pressure",
        "Insulin Level": "insulin"
    }
    with col_metric:
        selected_metric_display = st.selectbox("Select Metric to Track:", list(metrics_to_plot.keys()), key="metric_select")
        selected_metric_column = metrics_to_plot[selected_metric_display]

    st.subheader(f"{selected_metric_display} Over Time")
    chart_metric = alt.Chart(df_minh).mark_line(point=True, color='#007bff').encode( # Blue line
        x=alt.X('time_period:O', title=time_title, axis=alt.Axis(labelAngle=-45)),
        y=alt.Y(f'{selected_metric_column}:Q', title=selected_metric_display),
        tooltip=[
            alt.Tooltip('timestamp:T', title='Date'),
            alt.Tooltip(f'{selected_metric_column}:Q', title=selected_metric_display, format='.2f')
        ]
    ).properties(
        title=f'{selected_metric_display} Trend Over Time'
    ).interactive()
    st.altair_chart(chart_metric, use_container_width=True)

    st.subheader("💡 Personalized Recommendations Based on Risk Trend:")
    if len(df_minh) >= 2:
        latest_risk = df_minh['risk_probability'].iloc[-1]
        previous_risk = df_minh['risk_probability'].iloc[-2]

        if latest_risk < previous_risk:
            st.success(
                """
                **Great News! Your risk probability has shown a positive trend, decreasing in the most recent assessment.** Keep up the excellent work!
                
                **To reinforce these healthy habits:**
                * **Balanced Diet:** Continue prioritizing whole grains, fruits, vegetables, and lean proteins. Limit sugary drinks and processed foods.
                * **Regular Activity:** Aim for consistent moderate-intensity physical activity.
                * **Quality Sleep:** Prioritize 7-9 hours of restful sleep each night.
                * **Stress Management:** Keep practicing relaxation techniques.
                * **Hydration:** Drink plenty of water daily.
                
                Remember to consult your healthcare provider for ongoing guidance.
                """
            )
        elif latest_risk > previous_risk:
            st.warning(
                """
                **Important! Your risk probability has increased in the recent assessment.** It's crucial to take proactive steps now.
                
                **Consider these actions:**
                * **Diet Review:** Re-evaluate your diet to reduce sugar, unhealthy fats, and processed foods. A nutritionist's advice could be beneficial.
                * **Boost Activity:** Gradually increase your physical activity. Find enjoyable ways to be active daily.
                * **Glucose Monitoring:** If advised, track your blood glucose levels and discuss patterns with your doctor.
                * **Stress Reduction:** Implement more stress-reducing activities into your routine.
                * **Doctor's Visit:** Schedule an appointment with your healthcare provider to discuss these changes and explore further evaluation or management.
                """
            )
        else:
            st.info(
                """
                **Your risk probability has remained relatively stable in the recent assessment.** Consistency is key!
                
                **Continue to be diligent with your healthy lifestyle:**
                * **Consistent Diet:** Maintain your balanced diet for stable blood sugar.
                * **Maintain Exercise:** Keep up your regular physical activity routine.
                * **Body Awareness:** Pay attention to any new symptoms and discuss them with your doctor.
                * **Regular Check-ups:** Routine visits are essential for ongoing monitoring and prevention.
                """
            )
    else:
        st.info("Not enough assessment data to determine a risk trend. Perform more assessments over time to gain better insights.")


def articles_page():
    st.header("📚 Diabetes Prevention Articles & Resources")
    st.markdown("Empower yourself with knowledge! Read our curated articles on how to prevent and manage diabetes effectively.")
    articles = {
        "Understanding Type 2 Diabetes: A Comprehensive Overview": {
            "icon": "📖",
            "content": """
            Type 2 diabetes mellitus (T2DM) is a complex metabolic disorder characterized by hyperglycemia resulting from defects in insulin secretion, insulin action, or both. It is a progressive condition that can lead to serious health complications if not managed effectively.

            **The Role of Insulin:** Insulin, a hormone produced by the pancreas, plays a crucial role in regulating blood glucose levels. It allows glucose from the bloodstream to enter cells, where it can be used for energy. In T2DM, the body either doesn't produce enough insulin (insulin deficiency) or the cells become resistant to the insulin that is produced (insulin resistance).

            **Risk Factors:** Several factors can increase the risk of developing T2DM, including:
            * **Obesity and Overweight:** Excess body weight, particularly abdominal fat, is a major risk factor.
            * **Family History:** Having a close relative with T2DM increases your likelihood of developing the condition.
            * **Age:** The risk of T2DM increases with age, particularly after 45.
            * **Physical Inactivity:** Lack of regular exercise contributes to insulin resistance.
            * **Unhealthy Diet:** A diet high in sugary drinks, processed foods, and unhealthy fats can increase the risk.
            * **Gestational Diabetes:** Women who had gestational diabetes during pregnancy have a higher risk of developing T2DM later in life.
            * **Certain Ethnicities:** Some ethnic groups, such as African Americans, Hispanic/Latino Americans, American Indians, and Asian Americans, have a higher prevalence of T2DM.

            **Symptoms:** The onset of T2DM can be gradual, and many people may not experience noticeable symptoms in the early stages. However, some common symptoms include:
            * Increased thirst (polydipsia)
            * Frequent urination (polyuria)
            * Increased hunger (polyphagia)
            * Unexplained weight loss
            * Fatigue
            * Blurred vision
            * Slow-healing sores or frequent infections

            Early diagnosis and management are crucial to prevent or delay the long-term complications of T2DM, which can include heart disease, stroke, kidney disease, nerve damage (neuropathy), and eye damage (retinopathy).
        """},
        "The Cornerstone of Prevention: Lifestyle Modifications": {
            "icon": "🍎",
            "content": """
            Lifestyle modifications are the most effective strategies for preventing or delaying the onset of type 2 diabetes, especially in individuals at high risk. These changes focus on diet, physical activity, and weight management.

            **Dietary Strategies:**
            * **Emphasize Whole Foods:** Build your diet around whole, unprocessed foods such as fruits, vegetables, whole grains, and lean protein sources.
            * **Limit Sugary Drinks and Processed Foods:** These are often high in calories, unhealthy fats, and added sugars, contributing to weight gain and insulin resistance.
            * **Increase Fiber Intake:** Dietary fiber, found in fruits, vegetables, and whole grains, helps regulate blood sugar levels and promotes satiety.
            * **Choose Healthy Fats:** Opt for unsaturated fats found in avocados, nuts, seeds, and olive oil, while limiting saturated and trans fats.
            * **Control Portion Sizes:** Being mindful of how much you eat can help manage calorie intake and prevent weight gain.

            **Physical Activity Recommendations:**
            * **Aim for Regular Exercise:** Engage in at least 150 minutes of moderate-intensity aerobic activity per week, such as brisk walking, cycling, or swimming.
            * **Include Strength Training:** Incorporate strength training exercises at least two days a week to build muscle mass, which can improve insulin sensitivity.
            * **Reduce Sedentary Time:** Break up long periods of sitting with short bursts of activity throughout the day.

            **Weight Management:**
            * **Achieve and Maintain a Healthy Weight:** Losing even a small amount of weight (5-7% of body weight) can significantly reduce the risk of developing T2DM.
            * **Set Realistic Goals:** Focus on gradual and sustainable weight loss through a combination of diet and exercise.
            * **Seek Support:** If you are struggling to lose weight, consider seeking guidance from a healthcare professional or a registered dietitian.

            Adopting these lifestyle modifications can have a profound impact on reducing your risk of type 2 diabetes and improving your overall health.
        """},
        "The Power of Physical Activity: Preventing Diabetes and Improving Health": {
            "icon": "🏃‍♀️",
            "content": """
            Regular physical activity is a cornerstone of diabetes prevention and offers a wide range of health benefits beyond blood sugar control.

            **Mechanisms of Action:**
            * **Improved Insulin Sensitivity:** Exercise makes your body's cells more responsive to insulin, allowing glucose to enter cells more effectively and lowering blood sugar levels.
            * **Weight Management:** Physical activity helps burn calories, contributing to weight loss and the maintenance of a healthy weight, which reduces insulin resistance.
            * **Lower Blood Sugar Levels:** During and after exercise, your muscles use glucose for energy, helping to lower blood sugar levels.
            * **Reduced Cardiovascular Risk:** Regular exercise improves heart health, lowers blood pressure and cholesterol levels, and reduces the risk of heart disease and stroke, common complications of diabetes.
            * **Increased Muscle Mass:** Strength training helps build muscle, which is more metabolically active than fat tissue, further improving glucose utilization.

            **Types of Exercise:**
            * **Aerobic Exercise:** Activities that get your heart rate up, such as brisk walking, running, swimming, cycling, dancing, and hiking. Aim for at least 150 minutes of moderate-intensity or 75 minutes of vigorous-intensity aerobic activity per week, or a combination of both.
            * **Strength Training:** Exercises that work your major muscle groups, such as lifting weights, using resistance bands, or doing bodyweight exercises. Aim for at least two days a week.
            * **Flexibility and Balance Exercises:** Activities like stretching and yoga can improve flexibility and balance, reducing the risk of falls, which is particularly important for older adults.

            **Making Exercise a Habit:**
            * **Start Slowly:** If you are new to exercise, begin with short, low-intensity workouts and gradually increase the duration and intensity.
            * **Find Activities You Enjoy:** Choosing activities you like will make it easier to stick with a regular exercise routine.
            * **Make it Part of Your Routine:** Schedule exercise into your day like any other important appointment.
            * **Find an Exercise Buddy:** Exercising with a friend can provide motivation and accountability.

            Incorporating regular physical activity into your lifestyle is a powerful tool for preventing diabetes and improving your overall well-being.
        """},
        "Nutrition and Diabetes Prevention: Fueling Your Body the Right Way": {
            "icon": "🥗",
            "content": """
            A well-balanced and nutritious diet plays a vital role in preventing type 2 diabetes by helping to maintain a healthy weight, regulate blood sugar levels, and improve insulin sensitivity.

            **Key Dietary Principles:**
            * **Prioritize Whole Grains:** Choose whole grains like brown rice, quinoa, oats, and whole-wheat bread over refined grains, which are digested quickly and can cause rapid blood sugar spikes.
            * **Load Up on Fruits and Vegetables:** These are rich in fiber, vitamins, minerals, and antioxidants, and are generally low in calories. Aim for a variety of colors to get a wide range of nutrients.
            * **Choose Lean Protein Sources:** Opt for lean protein sources such as fish, poultry without skin, beans, lentils, and tofu. Limit red and processed meats.
            * **Incorporate Healthy Fats:** Include sources of unsaturated fats like avocados, nuts, seeds, and olive oil. Limit saturated and trans fats found in processed foods and fatty meats.
            * **Limit Added Sugars:** Reduce your intake of sugary drinks (soda, juice), candy, pastries, and other foods high in added sugars, which contribute to weight gain and increase diabetes risk.
            * **Control Portion Sizes:** Be mindful of how much you are eating, even of healthy foods, to manage calorie intake.
            * **Stay Hydrated:** Drink plenty of water throughout the day. Avoid sugary beverages.

            **Practical Tips for Healthy Eating:**
            * **Plan Your Meals:** Planning ahead can help you make healthier choices and avoid impulsive, unhealthy options.
            * **Read Food Labels:** Pay attention to serving sizes, calories, sugar content, and fat content.
            * **Cook at Home More Often:** This gives you more control over the ingredients and preparation methods.
            * **Be Mindful While Eating:** Pay attention to your hunger and fullness cues, and eat slowly.

            Making sustainable changes to your eating habits is a crucial step in preventing type 2 diabetes and promoting long-term health.
        """},
        "The Importance of Sleep and Stress Management in Diabetes Prevention": {
            "icon": "😴",
            "content": """
            While diet and exercise are often the primary focus of diabetes prevention, adequate sleep and effective stress management also play significant roles in regulating blood sugar levels and overall metabolic health.

            **The Impact of Sleep:**
            * **Insulin Sensitivity:** Chronic sleep deprivation can lead to insulin resistance, making it harder for your body to use insulin effectively and increasing blood sugar levels.
            * **Hormone Regulation:** Lack of sleep can disrupt the balance of hormones that regulate appetite and metabolism, potentially leading to increased hunger, weight gain, and an increased risk of diabetes.
            * **Glucose Metabolism:** Studies have shown that insufficient sleep can impair glucose tolerance, meaning the body is less efficient at processing glucose.

            **Tips for Better Sleep:**
            * **Establish a Regular Sleep Schedule:** Go to bed and wake up around the same time each day, even on weekends.
            * **Create a Relaxing Bedtime Routine:** Wind down before bed with activities like reading, taking a warm bath, or listening to calming music.
            * **Optimize Your Sleep Environment:** Make sure your bedroom is dark, quiet, and cool.
            * **Avoid Caffeine and Alcohol Before Bed:** These substances can interfere with sleep.
            * **Limit Screen Time Before Bed:** The blue light emitted from electronic devices can suppress melatonin production, making it harder to fall asleep.

            **The Role of Stress:**
            * **Stress Hormones:** When you are stressed, your body releases hormones like cortisol and adrenaline, which can raise blood sugar levels.
            * **Unhealthy Coping Mechanisms:** Chronic stress can lead to unhealthy coping behaviors like overeating, choosing unhealthy foods, and reducing physical activity, all of which increase diabetes risk.

            **Effective Stress Management Techniques:**
            * **Regular Exercise:** Physical activity is a great way to relieve stress and improve mood.
            * **Mindfulness and Meditation:** These practices can help you focus on the present moment and reduce feelings of stress and anxiety.
            * **Deep Breathing Exercises:** Simple breathing techniques can help calm your nervous system.
            * **Spending Time on Hobbies:** Engaging in enjoyable activities can help you relax and reduce stress.
            * **Building a Strong Social Support Network:** Connecting with friends and family can provide emotional support during stressful times.
            * **Getting Enough Sleep:** As mentioned earlier, adequate sleep is crucial for managing stress.

            Prioritizing good sleep habits and developing effective stress management techniques are important components of a comprehensive diabetes prevention strategy.
        """},
    }
    
    # Create an intuitive grid for articles using columns
    cols = st.columns(3)
    for i, (title, data) in enumerate(articles.items()):
        with cols[i % 3]:
            with st.container(border=True): # Use container for visual separation
                st.markdown(f"#### {data['icon']} {title}")
                # Use expander to hide long content, improving initial readability
                with st.expander("Read More"):
                    st.markdown(data['content'])
    
    st.markdown("---")
    st.info("💡 **Disclaimer:** The information provided in these articles is for general knowledge and informational purposes only, and does not constitute medical advice. Always consult with a qualified healthcare professional for any health concerns or before making any decisions related to your health or treatment.")


def about_page():
    st.header("ℹ️ About Our Diabetes Risk Prediction & Prevention Hub")
    st.markdown("Learn about the vision, mission, and values driving our platform.")

    st.subheader("🌟 Our Vision")
    st.markdown(
        """
        To empower individuals with knowledge and tools to proactively manage their health,
        significantly reducing the prevalence and impact of type 2 diabetes in our community
        and beyond. We envision a future where informed lifestyle choices lead to healthier,
        longer lives, free from the burden of preventable chronic diseases.
        """
    )

    st.subheader("🎯 Our Mission")
    st.markdown(
        """
        Our mission is to provide an accessible, user-friendly platform that combines
        personalized diabetes risk assessment with comprehensive educational resources.
        We strive to:
        -   Offer a reliable initial assessment of type 2 diabetes risk based on individual health data.
        -   Deliver clear, evidence-based information on diabetes prevention and management.
        -   Encourage proactive engagement in healthy lifestyle modifications.
        -   Foster a sense of empowerment and informed decision-making regarding personal health.
        """
    )

    st.subheader("💖 Our Key Values")
    st.markdown(
        """
        We are guided by the following core values in everything we do:

        -   **Empowerment:** We believe in equipping individuals with the knowledge and tools they need to take control of their health journey.
        -   **Accessibility:** We are committed to making our platform and resources easily available to everyone, regardless of their background or technical expertise.
        -   **Reliability:** We strive to provide accurate risk assessments based on established models and present information that is grounded in scientific evidence.
        -   **Education:** We are passionate about delivering clear, understandable, and actionable information to promote health literacy.
        -   **User-Centricity:** We prioritize the needs and experience of our users, continuously seeking to improve and enhance our platform.
        -   **Privacy and Security:** We are dedicated to protecting the privacy and security of user data with the utmost care and responsibility.
        -   **Continuous Improvement:** We are committed to ongoing learning and development, constantly seeking ways to enhance the accuracy, functionality, and value of our hub.

        We believe that by adhering to these values, we can make a meaningful difference in the lives of individuals and contribute to a healthier future.
    """
    )

# --- Discussion Forum Functionality ---
def display_posts():
    if 'posts' not in st.session_state:
        st.session_state['posts'] = [
            {'title': '🌟 Welcome to the Community!', 'content': "Hi everyone, I'm new here and looking forward to learning more about diabetes prevention and sharing experiences. What are your top tips for staying healthy?", 'comments': ["Welcome! Start with small changes.", "Consistency is key!", "Don't underestimate sleep."]},
            {'title': '🍏 Question about Low-Sugar Snacks', 'content': "I'm trying to reduce my sugar intake. What are some good, easy low-sugar snack options you'd recommend?", 'comments': ["Nuts and seeds are a good choice!", "Greek yogurt with berries (in moderation).", "Vegetable sticks with hummus.", "Hard-boiled eggs are great for protein."]},
            {'title': '💪 Motivation for Exercise', 'content': "Does anyone have advice for staying motivated to exercise regularly, especially when life gets busy?", 'comments': ["Find an exercise buddy!", "Set small, achievable goals.", "Try different activities until you find something you enjoy.", "Schedule it like an important appointment."]}
        ]
    
    st.subheader("💬 Active Discussions")
    for i, post in enumerate(st.session_state['posts']):
        with st.container(border=True):
            st.markdown(f"### {post['title']}")
            st.markdown(post['content'])
            
            # Use columns for "Add Comment" button and comment count
            col_comment_btn, col_comment_count = st.columns([0.7, 0.3])

            with col_comment_count:
                comment_count = len(post.get('comments', []))
                st.markdown(f"**Comments:** {comment_count}")

            with col_comment_btn:
                # Use a unique key for each expander to manage comments
                with st.expander(f"View/Add Comments for '{post['title'][:30]}...'"):
                    if 'comments' in post and post['comments']:
                        for comment in post['comments']:
                            st.markdown(f"**•** __{comment}__")
                    else:
                        st.info("No comments yet. Be the first to add one!")

                    with st.form(key=f'comment_form_{i}'):
                        comment_text = st.text_input("Your comment:", key=f'comment_text_input_{i}')
                        submit_comment_btn = st.form_submit_button("Post Comment", type="primary")
                        if submit_comment_btn:
                            if comment_text:
                                if 'comments' not in st.session_state['posts'][i]:
                                    st.session_state['posts'][i]['comments'] = []
                                st.session_state['posts'][i]['comments'].append(comment_text)
                                st.success("Your comment has been posted!")
                                st.rerun()
                            else:
                                st.warning("Please type something to post a comment.")

def create_new_post():
    st.subheader("📝 Start a New Discussion")
    with st.form(key='new_post_form', clear_on_submit=True):
        title = st.text_input("Post Title (e.g., 'My Experience with Low-Carb Diet')", key='new_post_title')
        content = st.text_area("Your Post Content (share your thoughts or questions here)", key='new_post_content')
        col_submit_post_btn, _ = st.columns([0.4, 0.6])
        with col_submit_post_btn:
            if st.form_submit_button("Create New Post", type="primary", use_container_width=True):
                if title and content:
                    new_post = {'title': title, 'content': content, 'comments': []}
                    if 'posts' not in st.session_state:
                        st.session_state['posts'] = []
                    st.session_state['posts'].insert(0, new_post) # Add new post at the top
                    st.success("Your new post has been created!")
                    st.rerun() # Rerun to display the new post immediately
                else:
                    st.warning("Please enter both a title and content for your post.")
    st.markdown("---") # Separator

def discussion_forum_page():
    st.header("🗣️ Community Discussion Forum")
    st.markdown("Connect with others, share experiences, and ask questions about diabetes prevention and healthy living.")
    
    create_new_post()
    display_posts()

# --- Doctor Appointment Page ---
def doctor_appointment_page():
    st.header("🗓️ Doctor Appointment Booking & Management")
    st.markdown("Easily book and manage your healthcare appointments. This is a simulated booking system for demonstration purposes.")

    # Dummy data for doctors and hospitals
    doctors = [
        "Dr. Alice Smith (Endocrinologist)",
        "Dr. Bob Johnson (General Practitioner)",
        "Dr. Carol White (Nutritionist)",
        "Dr. David Green (Cardiologist)"
    ]
    hospitals = [
        "City General Hospital",
        "Diabetes Care Clinic",
        "Wellness Medical Center",
        "Community Health Hub"
    ]
    appointment_reasons = [
        "General Check-up",
        "Diabetes Management",
        "Diet Consultation",
        "Symptoms Review",
        "Follow-up",
        "Specialist Referral"
    ]

    st.subheader("➕ Book a New Appointment")
    with st.form(key='appointment_form', clear_on_submit=True):
        col_doc, col_hosp = st.columns(2)
        selected_doctor = col_doc.selectbox("Select Doctor:", doctors, key='doctor_select', help="Choose the specialist or general practitioner you wish to see.")
        selected_hospital = col_hosp.selectbox("Select Hospital/Clinic:", hospitals, key='hospital_select', help="Where would you like your appointment to be held?")
        
        col_date, col_time = st.columns(2)
        appointment_date = col_date.date_input("Select Date:", min_value=datetime.date.today(), key='date_input', help="Choose a date for your appointment.")
        appointment_time = col_time.time_input("Select Time:", datetime.time(9, 0), step=datetime.timedelta(minutes=30), key='time_input', help="Choose a preferred time slot (30-minute intervals).")
        
        selected_reason = st.selectbox("Reason for Appointment:", appointment_reasons, key='reason_select', help="What is the main reason for this visit?")
        additional_notes = st.text_area("Additional Notes (optional):", help="Provide any additional details that might be helpful for your doctor.", key='notes_area')

        submit_appointment = st.form_submit_button("✅ Book Appointment", type="primary", use_container_width=True)

        if submit_appointment:
            new_appointment = {
                "doctor": selected_doctor,
                "hospital": selected_hospital,
                "date": appointment_date,
                "time": appointment_time,
                "reason": selected_reason,
                "notes": additional_notes,
                "status": "Booked"
            }
            if 'appointments' not in st.session_state:
                st.session_state['appointments'] = []
            st.session_state['appointments'].append(new_appointment)
            st.success(f"**Success!** Your appointment with **{selected_doctor}** at **{selected_hospital}** on **{appointment_date.strftime('%Y-%m-%d')}** at **{appointment_time.strftime('%H:%M')}** has been booked! You'll receive a confirmation shortly. ✉️")
            st.balloons() # Add a little celebration
            st.rerun() # Rerun to update the displayed appointments

    st.markdown("---")
    st.subheader("📋 Your Booked Appointments")

    if 'appointments' in st.session_state and st.session_state['appointments']:
        # Sort appointments by date and time
        sorted_appointments = sorted(
            st.session_state['appointments'],
            key=lambda x: (x['date'], x['time'])
        )

        # Create a DataFrame for display purposes.
        display_data = []
        for appt in sorted_appointments:
            display_data.append({
                "Date": appt['date'].strftime('%Y-%m-%d'),
                "Time": appt['time'].strftime('%H:%M'),
                "Doctor": appt['doctor'],
                "Hospital": appt['hospital'],
                "Reason": appt['reason'],
                "Status": appt['status']
            })
        
        display_df = pd.DataFrame(display_data)

        st.dataframe(display_df, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.subheader("⚙️ Manage Existing Appointments")

        # Allow users to manage individual appointments
        for i, appt in enumerate(sorted_appointments):
            unique_key_prefix = f"appt_{i}"
            status_color = "green" if appt['status'] == "Booked" else ("red" if appt['status'] == "Cancelled" else "orange")
            with st.expander(f"**Appointment with {appt['doctor']} on {appt['date'].strftime('%Y-%m-%d')} at {appt['time'].strftime('%H:%M')}**"):
                st.markdown(f"**Doctor:** `{appt['doctor']}`")
                st.markdown(f"**Hospital:** `{appt['hospital']}`")
                st.markdown(f"**Date:** `{appt['date'].strftime('%Y-%m-%d')}`")
                st.markdown(f"**Time:** `{appt['time'].strftime('%H:%M')}`")
                st.markdown(f"**Reason:** `{appt['reason']}`")
                if appt['notes']:
                    st.markdown(f"**Notes:** `{appt['notes']}`")
                st.markdown(f"**Status:** <span style='color:{status_color}; font-weight:bold;'>{appt['status']}</span>", unsafe_allow_html=True)

                # Add Cancel button
                if appt['status'] == "Booked":
                    if st.button(f"🚫 Cancel This Appointment", key=f"{unique_key_prefix}_cancel"):
                        # Find the actual appointment object in the session state list and update its status
                        for session_appt in st.session_state['appointments']:
                            if session_appt['doctor'] == appt['doctor'] and \
                               session_appt['hospital'] == appt['hospital'] and \
                               session_appt['date'] == appt['date'] and \
                               session_appt['time'] == appt['time']:
                                session_appt['status'] = "Cancelled"
                                break
                        st.warning("Appointment has been cancelled.")
                        st.rerun() # Rerun to update status
                elif appt['status'] == "Cancelled":
                    st.info("This appointment was previously cancelled.")
                
    else:
        st.info("You currently have no booked appointments. Use the form above to schedule one!")

def main():
    st.title('💙 Diabetes Risk Prediction & Prevention Hub')
    st.markdown("Your partner in health management and diabetes prevention.")

    # Initialize session state for all persistent data
    if 'assessments' not in st.session_state:
        st.session_state['assessments'] = {}
    if 'show_minh_history' not in st.session_state:
        st.session_state['show_minh_history'] = False
    if 'posts' not in st.session_state:
        st.session_state['posts'] = [] # Will be populated by display_posts with defaults
    if 'appointments' not in st.session_state:
        st.session_state['appointments'] = []

    with st.sidebar:
        st.header("🌐 Navigation Menu")
        menu = {
            "Assess My Risk": "Risk Assessment",
            "Learn & Prevent": "Prevention Articles",
            "My Health Trends": "Assessment History",
            "Book a Doctor": "Doctor Appointment",
            "Community Forum": "Discussion Forum",
            "About This Hub": "About"
        }
        # Use a more descriptive selectbox for navigation
        choice_display = st.selectbox("Explore our features:", list(menu.keys()), key="main_menu_select")
        choice = menu[choice_display] # Map display text back to internal choice

        st.markdown("---")
        st.markdown(
            """
            <div style='text-align: center; font-size: 0.9em; color: #555;'>
                Developed with ❤️ by Healthcare & Coding Enthusiasts<br>
                Empowering you for better health.
            </div>
            """, unsafe_allow_html=True
        )

    # Display selected page
    if choice == "Risk Assessment":
        risk_assessment_page()
    elif choice == "Prevention Articles":
        articles_page()
    elif choice == "Assessment History":
        assessment_history_page()
    elif choice == "Doctor Appointment":
        doctor_appointment_page()
    elif choice == "Discussion Forum":
        discussion_forum_page()
    elif choice == "About":
        about_page()

if __name__ == '__main__':
    main()
