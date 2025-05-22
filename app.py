import numpy as np
import joblib
import streamlit as st
import pandas as pd
import datetime
import altair as alt

# --- Load the Model ---
try:
    model_path = 'trained_model.joblib'
    loaded_model = joblib.load(model_path)
except FileNotFoundError:
    st.error("Error: Trained model file not found. Please check the path.")
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

def display_risk_interpretation(risk_level):
    if risk_level == "Low":
        st.success("Your estimated risk is relatively low. Continue to maintain a healthy lifestyle.")
    elif risk_level == "Moderate":
        st.warning("Your estimated risk is moderate. Consider lifestyle adjustments and regular checkups.")
    elif risk_level == "Elevated":
        st.error("Your estimated risk is elevated. It is strongly recommended to consult with a healthcare professional for further evaluation.")
    else:
        st.info("Risk assessment unavailable.")

def load_assessment_history(user_id):
    return st.session_state.get('assessments', {}).get(user_id, [])

# --- Risk Assessment Page ---
def risk_assessment_page():
    st.header("Diabetes Risk Assessment")
    st.subheader("Enter Your Health Information")

    # Initialize session state for inputs if they don't exist
    if 'input_age' not in st.session_state: st.session_state['input_age'] = None
    if 'input_pregnancies' not in st.session_state: st.session_state['input_pregnancies'] = None
    if 'input_glucose' not in st.session_state: st.session_state['input_glucose'] = None
    if 'input_blood_pressure' not in st.session_state: st.session_state['input_blood_pressure'] = None
    if 'input_skin_thickness' not in st.session_state: st.session_state['input_skin_thickness'] = None
    if 'input_insulin' not in st.session_state: st.session_state['input_insulin'] = None
    if 'input_weight' not in st.session_state: st.session_state['input_weight'] = None
    if 'input_height' not in st.session_state: st.session_state['input_height'] = None
    if 'input_diabetes_pedigree_function' not in st.session_state: st.session_state['input_diabetes_pedigree_function'] = None

    # Create tabs for Basic and Advanced Information
    tab1, tab2 = st.tabs(["Basic Information", "Advanced Medical Information"])

    with tab1:
        st.subheader("Personal Details")
        col1_basic, col2_basic = st.columns(2)
        age_input = col1_basic.number_input('Age (years)', min_value=0, max_value=120, value=st.session_state['input_age'], key='age_input_key')
        pregnancies_input = col1_basic.number_input('Number of Pregnancies', min_value=0, max_value=20, value=st.session_state['input_pregnancies'], key='pregnancies_input_key')

        weight_input = col2_basic.number_input('Weight (kg)', min_value=20, max_value=300, value=st.session_state['input_weight'], key='weight_input_key')
        height_input = col2_basic.number_input('Height (meters)', min_value=0.5, max_value=3.0, value=st.session_state['input_height'], step=0.01, key='height_input_key')

        # Store the current input values in session state
        st.session_state['input_age'] = age_input
        st.session_state['input_pregnancies'] = pregnancies_input
        st.session_state['input_weight'] = weight_input
        st.session_state['input_height'] = height_input

        # BMI calculation will happen after inputs are finalized (potentially with defaults)
        bmi = calculate_bmi(
            st.session_state['input_weight'] if st.session_state['input_weight'] is not None else DEFAULT_WEIGHT,
            st.session_state['input_height'] if st.session_state['input_height'] is not None else DEFAULT_HEIGHT
        )
        st.metric("Calculated BMI", f"{bmi:.2f}")
        st.session_state['BMI'] = bmi # Store BMI in session state


    with tab2:
        st.subheader("Clinical Parameters")
        col1_adv, col2_adv = st.columns(2)
        glucose_input = col1_adv.number_input('Glucose Level (mg/dL)', min_value=0, max_value=300, value=st.session_state['input_glucose'], key='glucose_input_key')
        blood_pressure_input = col1_adv.number_input('Blood Pressure (mmHg)', min_value=0, max_value=200, value=st.session_state['input_blood_pressure'], key='blood_pressure_input_key')
        skin_thickness_input = col2_adv.number_input('Skin Thickness (mm)', min_value=0, max_value=100, value=st.session_state['input_skin_thickness'], key='skin_thickness_input_key')
        insulin_input = col2_adv.number_input('Insulin Level (mu U/mL)', min_value=0, max_value=850, value=st.session_state['input_insulin'], key='insulin_input_key')
        diabetes_pedigree_function_input = col1_adv.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=3.0, value=st.session_state['input_diabetes_pedigree_function'], step=0.001, key='dpf_input_key')

        # Store the current input values in session state
        st.session_state['input_glucose'] = glucose_input
        st.session_state['input_blood_pressure'] = blood_pressure_input
        st.session_state['input_skin_thickness'] = skin_thickness_input
        st.session_state['input_insulin'] = insulin_input
        st.session_state['input_diabetes_pedigree_function'] = diabetes_pedigree_function_input

    st.markdown("---") # Separator between inputs and button

    if st.button("Assess Risk"):
        if loaded_model:
            # Use actual input values, or default if None (left blank by user)
            final_age = st.session_state['input_age'] if st.session_state['input_age'] is not None else DEFAULT_AGE
            final_pregnancies = st.session_state['input_pregnancies'] if st.session_state['input_pregnancies'] is not None else DEFAULT_PREGNANCIES
            final_glucose = st.session_state['input_glucose'] if st.session_state['input_glucose'] is not None else DEFAULT_GLUCOSE
            final_blood_pressure = st.session_state['input_blood_pressure'] if st.session_state['input_blood_pressure'] is not None else DEFAULT_BLOODPRESSURE
            final_skin_thickness = st.session_state['input_skin_thickness'] if st.session_state['input_skin_thickness'] is not None else DEFAULT_SKINTHICKNESS
            final_insulin = st.session_state['input_insulin'] if st.session_state['input_insulin'] is not None else DEFAULT_INSULIN
            final_weight = st.session_state['input_weight'] if st.session_state['input_weight'] is not None else DEFAULT_WEIGHT
            final_height = st.session_state['input_height'] if st.session_state['input_height'] is not None else DEFAULT_HEIGHT
            final_diabetes_pedigree_function = st.session_state['input_diabetes_pedigree_function'] if st.session_state['input_diabetes_pedigree_function'] is not None else DEFAULT_DIABETESPEDIGREEFUNCTION

            # Recalculate BMI with potentially defaulted weight/height for the prediction input
            final_bmi = calculate_bmi(final_weight, final_height)

            input_data = [final_pregnancies, final_glucose, final_blood_pressure, final_skin_thickness, final_insulin, final_bmi, final_diabetes_pedigree_function, final_age]
            risk_probability = diabetes_prediction_proba(input_data, loaded_model)
            risk_level = get_risk_level(risk_probability)

            st.subheader(f"Estimated Diabetes Risk: {risk_level} ({risk_probability * 100:.2f}%)")
            display_risk_interpretation(risk_level)
        else:
            st.error("Model not loaded. Cannot assess risk.")

def assessment_history_page():
    st.header("Assessment History")
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

    st.subheader("View Assessment Data By:")
    view_by = st.radio("", ["Month", "Week"], horizontal=True)

    if view_by == "Month":
        df_minh['time_period'] = df_minh['timestamp'].dt.to_period('M').astype(str)
        time_title = 'Month'
    else:
        df_minh['time_period'] = df_minh['timestamp'].dt.isocalendar().week.astype(str) + '-' + df_minh['timestamp'].dt.year.astype(str)
        time_title = 'Week'

    metrics_to_plot = ["risk_probability", "bmi", "glucose", "blood_pressure", "insulin"]
    selected_metric = st.selectbox("Select metric to view:", metrics_to_plot)
    metric_title = selected_metric.replace("_", " ").title()

    st.subheader(f"{metric_title} Over Time (Last 12 Months)")
    chart_metric = alt.Chart(df_minh).mark_line(point=True).encode(
        x=alt.X('time_period:O', title=time_title),
        y=alt.Y(f'{selected_metric}:Q', title=metric_title),
        tooltip=['timestamp:T', f'{selected_metric}:Q']
    ).properties(
        title=f'{metric_title} Trend'
    ).interactive()
    st.altair_chart(chart_metric, use_container_width=True)

    st.subheader("Personalized Recommendations Based on Risk Trend:")
    if len(df_minh) >= 2:
        latest_risk = df_minh['risk_probability'].iloc[-1]
        previous_risk = df_minh['risk_probability'].iloc[-2]

        if latest_risk < previous_risk:
            st.success(
                """
                Your risk probability has shown a positive trend, decreasing in the most recent assessment.
                Continue to reinforce these healthy habits:
                - **Maintain a balanced diet:** Focus on whole grains, fruits, vegetables, and lean proteins. Limit sugary drinks and processed foods.
                - **Engage in regular physical activity:** Aim for at least 150 minutes of moderate-intensity or 75 minutes of vigorous-intensity exercise per week.
                - **Ensure adequate sleep:** Prioritize 7-9 hours of quality sleep each night.
                - **Manage stress effectively:** Practice relaxation techniques like mindfulness or yoga.
                - **Stay hydrated:** Drink plenty of water throughout the day.
                Continue to monitor your health and consult with your healthcare provider for ongoing guidance.
                """
            )
        elif latest_risk > previous_risk:
            st.warning(
                """
                Your risk probability has unfortunately increased in the recent assessment.
                It's important to take proactive steps:
                - **Review your current diet:** Identify areas where you can reduce sugar, unhealthy fats, and processed foods. Consider consulting a nutritionist for personalized dietary advice.
                - **Increase physical activity:** If you're not currently active, start gradually and aim for regular exercise. Explore activities you enjoy to make it sustainable.
                - **Monitor your blood glucose levels:** If you have a home glucose meter, track your readings and discuss any patterns with your doctor.
                - **Assess stress levels:** High stress can impact blood sugar. Implement stress-reducing activities.
                - **Schedule a check-up:** Make an appointment with your healthcare provider to discuss these changes and explore further evaluation or management strategies.
                """
            )
        else:
            st.info(
                """
                Your risk probability has remained relatively stable in the recent assessment.
                Continue to be diligent with your healthy lifestyle:
                - **Stay consistent with your current diet:** Ensure it remains balanced and supports healthy blood sugar levels.
                - **Maintain your exercise routine:** Regular physical activity is key for long-term health.
                - **Pay attention to any changes in your body:** Be aware of any new symptoms or concerns and discuss them with your doctor.
                - **Schedule regular check-ups:** Routine visits with your healthcare provider are essential for ongoing monitoring and prevention.
                """
            )
    else:
        st.info("Not enough assessment data to determine a risk trend. More assessments over time will provide better insights.")

def articles_page():
    st.header("Diabetes Prevention Articles")
    articles = {
        "Understanding Type 2 Diabetes: A Comprehensive Overview": """
            Type 2 diabetes mellitus (T2DM) is a complex metabolic disorder characterized by hyperglycemia resulting from defects in insulin secretion, insulin action, or both. It is a progressive condition that can lead to serious health complications if not managed effectively.

            **The Role of Insulin:** Insulin, a hormone produced by the pancreas, plays a crucial role in regulating blood glucose levels. It allows glucose from the bloodstream to enter cells, where it can be used for energy. In T2DM, the body either doesn't produce enough insulin (insulin deficiency) or the cells become resistant to the insulin that is produced (insulin resistance).

            **Risk Factors:** Several factors can increase the risk of developing T2DM, including:
            - **Obesity and Overweight:** Excess body weight, particularly abdominal fat, is a major risk factor.
            - **Family History:** Having a close relative with T2DM increases your likelihood of developing the condition.
            - **Age:** The risk of T2DM increases with age, particularly after 45.
            - **Physical Inactivity:** Lack of regular exercise contributes to insulin resistance.
            - **Unhealthy Diet:** A diet high in sugary drinks, processed foods, and unhealthy fats can increase the risk.
            - **Gestational Diabetes:** Women who had gestational diabetes during pregnancy have a higher risk of developing T2DM later in life.
            - **Certain Ethnicities:** Some ethnic groups, such as African Americans, Hispanic/Latino Americans, American Indians, and Asian Americans, have a higher prevalence of T2DM.

            **Symptoms:** The onset of T2DM can be gradual, and many people may not experience noticeable symptoms in the early stages. However, some common symptoms include:
            - Increased thirst (polydipsia)
            - Frequent urination (polyuria)
            - Increased hunger (polyphagia)
            - Unexplained weight loss
            - Fatigue
            - Blurred vision
            - Slow-healing sores or frequent infections

            Early diagnosis and management are crucial to prevent or delay the long-term complications of T2DM, which can include heart disease, stroke, kidney disease, nerve damage (neuropathy), and eye damage (retinopathy).
        """,
        "The Cornerstone of Prevention: Lifestyle Modifications": """
            Lifestyle modifications are the most effective strategies for preventing or delaying the onset of type 2 diabetes, especially in individuals at high risk. These changes focus on diet, physical activity, and weight management.

            **Dietary Strategies:**
            - **Emphasize Whole Foods:** Build your diet around whole, unprocessed foods such as fruits, vegetables, whole grains, and lean protein sources.
            - **Limit Sugary Drinks and Processed Foods:** These are often high in calories, unhealthy fats, and added sugars, contributing to weight gain and insulin resistance.
            - **Increase Fiber Intake:** Dietary fiber, found in fruits, vegetables, and whole grains, helps regulate blood sugar levels and promotes satiety.
            - **Choose Healthy Fats:** Opt for unsaturated fats found in avocados, nuts, seeds, and olive oil, while limiting saturated and trans fats.
            - **Control Portion Sizes:** Being mindful of how much you eat can help manage calorie intake and prevent weight gain.

            **Physical Activity Recommendations:**
            - **Aim for Regular Exercise:** Engage in at least 150 minutes of moderate-intensity aerobic activity per week, such as brisk walking, cycling, or swimming.
            - **Include Strength Training:** Incorporate strength training exercises at least two days a week to build muscle mass, which can improve insulin sensitivity.
            - **Reduce Sedentary Time:** Break up long periods of sitting with short bursts of activity throughout the day.

            **Weight Management:**
            - **Achieve and Maintain a Healthy Weight:** Losing even a small amount of weight (5-7% of body weight) can significantly reduce the risk of developing T2DM.
            - **Set Realistic Goals:** Focus on gradual and sustainable weight loss through a combination of diet and exercise.
            - **Seek Support:** If you are struggling to lose weight, consider seeking guidance from a healthcare professional or a registered dietitian.

            Adopting these lifestyle modifications can have a profound impact on reducing your risk of type 2 diabetes and improving your overall health.
        """,
        "The Power of Physical Activity: Preventing Diabetes and Improving Health": """
            Regular physical activity is a cornerstone of diabetes prevention and offers a wide range of health benefits beyond blood sugar control.

            **Mechanisms of Action:**
            - **Improved Insulin Sensitivity:** Exercise makes your body's cells more responsive to insulin, allowing glucose to enter cells more effectively and lowering blood sugar levels.
            - **Weight Management:** Physical activity helps burn calories, contributing to weight loss and the maintenance of a healthy weight, which reduces insulin resistance.
            - **Lower Blood Sugar Levels:** During and after exercise, your muscles use glucose for energy, helping to lower blood sugar levels.
            - **Reduced Cardiovascular Risk:** Regular exercise improves heart health, lowers blood pressure and cholesterol levels, and reduces the risk of heart disease and stroke, common complications of diabetes.
            - **Increased Muscle Mass:** Strength training helps build muscle, which is more metabolically active than fat tissue, further improving glucose utilization.

            **Types of Exercise:**
            - **Aerobic Exercise:** Activities that get your heart rate up, such as brisk walking, running, swimming, cycling, dancing, and hiking. Aim for at least 150 minutes of moderate-intensity or 75 minutes of vigorous-intensity aerobic activity per week, or a combination of both.
            - **Strength Training:** Exercises that work your major muscle groups, such as lifting weights, using resistance bands, or doing bodyweight exercises. Aim for at least two days a week.
            - **Flexibility and Balance Exercises:** Activities like stretching and yoga can improve flexibility and balance, reducing the risk of falls, which is particularly important for older adults.

            **Making Exercise a Habit:**
            - **Start Slowly:** If you are new to exercise, begin with short, low-intensity workouts and gradually increase the duration and intensity.
            - **Find Activities You Enjoy:** Choosing activities you like will make it easier to stick with a regular exercise routine.
            - **Make it Part of Your Routine:** Schedule exercise into your day like any other important appointment.
            - **Find an Exercise Buddy:** Exercising with a friend can provide motivation and accountability.

            Incorporating regular physical activity into your lifestyle is a powerful tool for preventing diabetes and improving your overall well-being.
        """,
        "Nutrition and Diabetes Prevention: Fueling Your Body the Right Way": """
            A well-balanced and nutritious diet plays a vital role in preventing type 2 diabetes by helping to maintain a healthy weight, regulate blood sugar levels, and improve insulin sensitivity.

            **Key Dietary Principles:**
            - **Prioritize Whole Grains:** Choose whole grains like brown rice, quinoa, oats, and whole-wheat bread over refined grains, which are digested quickly and can cause rapid blood sugar spikes.
            - **Load Up on Fruits and Vegetables:** These are rich in fiber, vitamins, minerals, and antioxidants, and are generally low in calories. Aim for a variety of colors to get a wide range of nutrients.
            - **Choose Lean Protein Sources:** Opt for lean protein sources such as fish, poultry without skin, beans, lentils, and tofu. Limit red and processed meats.
            - **Incorporate Healthy Fats:** Include sources of unsaturated fats like avocados, nuts, seeds, and olive oil. Limit saturated and trans fats found in processed foods and fatty meats.
            - **Limit Added Sugars:** Reduce your intake of sugary drinks (soda, juice), candy, pastries, and other foods high in added sugars, which contribute to weight gain and increase diabetes risk.
            - **Control Portion Sizes:** Be mindful of how much you are eating, even of healthy foods, to manage calorie intake.
            - **Stay Hydrated:** Drink plenty of water throughout the day. Avoid sugary beverages.

            **Practical Tips for Healthy Eating:**
            - **Plan Your Meals:** Planning ahead can help you make healthier choices and avoid impulsive, unhealthy options.
            - **Read Food Labels:** Pay attention to serving sizes, calories, sugar content, and fat content.
            - **Cook at Home More Often:** This gives you more control over the ingredients and preparation methods.
            - **Be Mindful While Eating:** Pay attention to your hunger and fullness cues, and eat slowly.

            Making sustainable changes to your eating habits is a crucial step in preventing type 2 diabetes and promoting long-term health.
        """,
        "The Importance of Sleep and Stress Management in Diabetes Prevention": """
            While diet and exercise are often the primary focus of diabetes prevention, adequate sleep and effective stress management also play significant roles in regulating blood sugar levels and overall metabolic health.

            **The Impact of Sleep:**
            - **Insulin Sensitivity:** Chronic sleep deprivation can lead to insulin resistance, making it harder for your body to use insulin effectively and increasing blood sugar levels.
            - **Hormone Regulation:** Lack of sleep can disrupt the balance of hormones that regulate appetite and metabolism, potentially leading to increased hunger, weight gain, and an increased risk of diabetes.
            - **Glucose Metabolism:** Studies have shown that insufficient sleep can impair glucose tolerance, meaning the body is less efficient at processing glucose.

            **Tips for Better Sleep:**
            - **Establish a Regular Sleep Schedule:** Go to bed and wake up around the same time each day, even on weekends.
            - **Create a Relaxing Bedtime Routine:** Wind down before bed with activities like reading, taking a warm bath, or listening to calming music.
            - **Optimize Your Sleep Environment:** Make sure your bedroom is dark, quiet, and cool.
            - **Avoid Caffeine and Alcohol Before Bed:** These substances can interfere with sleep.
            - **Limit Screen Time Before Bed:** The blue light emitted from electronic devices can suppress melatonin production, making it harder to fall asleep.

            **The Role of Stress:**
            - **Stress Hormones:** When you are stressed, your body releases hormones like cortisol and adrenaline, which can raise blood sugar levels.
            - **Unhealthy Coping Mechanisms:** Chronic stress can lead to unhealthy coping behaviors like overeating, choosing unhealthy foods, and reducing physical activity, all of which increase diabetes risk.

            **Effective Stress Management Techniques:**
            - **Regular Exercise:** Physical activity is a great way to relieve stress and improve mood.
            - **Mindfulness and Meditation:** These practices can help you focus on the present moment and reduce feelings of stress and anxiety.
            - **Deep Breathing Exercises:** Simple breathing techniques can help calm your nervous system.
            - **Spending Time on Hobbies:** Engaging in enjoyable activities can help you relax and reduce stress.
            - **Building a Strong Social Support Network:** Connecting with friends and family can provide emotional support during stressful times.
            - **Getting Enough Sleep:** As mentioned earlier, adequate sleep is crucial for managing stress.

            Prioritizing good sleep habits and developing effective stress management techniques are important components of a comprehensive diabetes prevention strategy.
        """,
    }
    selected_article = st.selectbox("Select an article to read:", list(articles.keys()))
    st.subheader(selected_article)
    st.markdown(articles[selected_article])

def about_page():
    st.header("About Our Diabetes Risk Prediction & Prevention Hub")

    st.subheader("Our Vision")
    st.markdown(
        """
        To empower individuals with knowledge and tools to proactively manage their health,
        significantly reducing the prevalence and impact of type 2 diabetes in our community
        and beyond. We envision a future where informed lifestyle choices lead to healthier,
        longer lives, free from the burden of preventable chronic diseases.
        """
    )

    st.subheader("Our Mission")
    st.markdown(
        """
        Our mission is to provide an accessible, user-friendly platform that combines
        personalized diabetes risk assessment with comprehensive educational resources.
        We strive to:
        - Offer a reliable initial assessment of type 2 diabetes risk based on individual health data.
        - Deliver clear, evidence-based information on diabetes prevention and management.
        - Encourage proactive engagement in healthy lifestyle modifications.
        - Foster a sense of empowerment and informed decision-making regarding personal health.
        """
    )

    st.subheader("Our Key Values")
    st.markdown(
        """
        We are guided by the following core values in everything we do:

        - **Empowerment:** We believe in equipping individuals with the knowledge and tools they need to take control of their health journey.
        - **Accessibility:** We are committed to making our platform and resources easily available to everyone, regardless of their background or technical expertise.
        - **Reliability:** We strive to provide accurate risk assessments based on established models and present information that is grounded in scientific evidence.
        - **Education:** We are passionate about delivering clear, understandable, and actionable information to promote health literacy.
        - **User-Centricity:** We prioritize the needs and experience of our users, continuously seeking to improve and enhance our platform.
        - **Privacy and Security:** We are dedicated to protecting the privacy and security of user data with the utmost care and responsibility.
        - **Continuous Improvement:** We are committed to ongoing learning and development, constantly seeking ways to enhance the accuracy, functionality, and value of our hub.

        We believe that by adhering to these values, we can make a meaningful difference in the lives of individuals and contribute to a healthier future.
    """
    )

# --- Discussion Forum Functionality ---
def display_posts():
    if 'posts' not in st.session_state:
        st.session_state['posts'] = [
            {'title': 'New to the Community!', 'content': "Hi everyone, I'm new here and looking forward to learning more about diabetes prevention."},
            {'title': 'Question about Diet', 'content': "What are some good low-sugar snack options?", 'comments': ["Nuts and seeds are a good choice!", "Greek yogurt with berries.", "Vegetable sticks with hummus."]},
            {'title': 'Exercise Tips', 'content': "Does anyone have advice for staying motivated to exercise regularly?"}
        ]
    for i, post in enumerate(st.session_state['posts']):
        st.subheader(f"{post['title']}")
        st.markdown(post['content'])
        if 'comments' in post:
            with st.expander("View Comments"):
                for comment in post['comments']:
                    st.markdown(f"> {comment}")
        with st.form(key=f'comment_form_{i}'):
            comment_text = st.text_input("Add a comment:")
            if st.form_submit_button("Submit Comment"):
                if 'comments' not in st.session_state['posts'][i]:
                    st.session_state['posts'][i]['comments'] = []
                st.session_state['posts'][i]['comments'].append(comment_text)
                st.rerun()

def create_new_post():
    with st.form(key='new_post_form'):
        title = st.text_input("Post Title:")
        content = st.text_area("Post Content:")
        if st.form_submit_button("Create Post"):
            if title and content:
                new_post = {'title': title, 'content': content}
                if 'posts' not in st.session_state:
                    st.session_state['posts'] = []
                st.session_state['posts'].insert(0, new_post)
                st.rerun()
            else:
                st.warning("Please enter both a title and content for your post.")

def discussion_forum_page():
    st.header("Community Discussion Forum")
    create_new_post()
    st.subheader("Existing Discussions")
    display_posts()

# --- Doctor Appointment Page ---
def doctor_appointment_page():
    st.header("Doctor Appointment Booking & Management")

    # Dummy data for doctors and hospitals
    doctors = ["Dr. Alice Smith (Endocrinologist)", "Dr. Bob Johnson (General Practitioner)", "Dr. Carol White (Nutritionist)"]
    hospitals = ["City General Hospital", "Diabetes Care Clinic", "Wellness Medical Center"]
    appointment_reasons = ["General Check-up", "Diabetes Management", "Diet Consultation", "Symptoms Review", "Follow-up"]

    st.subheader("Book a New Appointment")
    with st.form(key='appointment_form'):
        selected_doctor = st.selectbox("Select Doctor:", doctors, key='doctor_select')
        selected_hospital = st.selectbox("Select Hospital/Clinic:", hospitals, key='hospital_select')
        appointment_date = st.date_input("Select Date:", min_value=datetime.date.today(), key='date_input')
        appointment_time = st.time_input("Select Time:", datetime.time(9, 0), step=60 * 30, key='time_input') # 30 min steps
        selected_reason = st.selectbox("Reason for Appointment:", appointment_reasons, key='reason_select')
        additional_notes = st.text_area("Additional Notes (optional):", key='notes_area')

        submit_appointment = st.form_submit_button("Book Appointment")

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
            st.success(f"Appointment with {selected_doctor} at {selected_hospital} on {appointment_date.strftime('%Y-%m-%d')} at {appointment_time.strftime('%H:%M')} has been booked!")
            st.rerun() # Rerun to update the displayed appointments

    st.subheader("Your Booked Appointments")

    if 'appointments' in st.session_state and st.session_state['appointments']:
        # Sort appointments by date and time
        sorted_appointments = sorted(
            st.session_state['appointments'],
            key=lambda x: (x['date'], x['time'])
        )

        # Create a DataFrame for display purposes.
        # Format date and time *after* sorting and *before* creating the display DataFrame
        # to avoid the .dt accessor error.
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
        st.subheader("Manage Appointments")

        # Allow users to manage individual appointments
        for i, appt in enumerate(sorted_appointments):
            unique_key_prefix = f"appt_{i}"
            with st.expander(f"Appointment: {appt['doctor']} on {appt['date'].strftime('%Y-%m-%d')} at {appt['time'].strftime('%H:%M')}"):
                st.write(f"**Doctor:** {appt['doctor']}")
                st.write(f"**Hospital:** {appt['hospital']}")
                st.write(f"**Date:** {appt['date'].strftime('%Y-%m-%d')}")
                st.write(f"**Time:** {appt['time'].strftime('%H:%M')}")
                st.write(f"**Reason:** {appt['reason']}")
                if appt['notes']:
                    st.write(f"**Notes:** {appt['notes']}")
                st.write(f"**Status:** {appt['status']}")

                # Add Cancel button
                if appt['status'] == "Booked":
                    if st.button(f"Cancel Appointment", key=f"{unique_key_prefix}_cancel"):
                        # Find the actual appointment object in the session state list and update its status
                        # This is important because `sorted_appointments` is a copy.
                        for session_appt in st.session_state['appointments']:
                            if session_appt['doctor'] == appt['doctor'] and \
                               session_appt['hospital'] == appt['hospital'] and \
                               session_appt['date'] == appt['date'] and \
                               session_appt['time'] == appt['time']:
                                session_appt['status'] = "Cancelled"
                                break
                        st.success("Appointment cancelled.")
                        st.rerun() # Rerun to update status
                elif appt['status'] == "Cancelled":
                    st.info("This appointment has been cancelled.")
                
                # Note: For simplicity, I'm not adding an "Edit" feature as it's more complex
                # and might require re-validating time slots etc.
    else:
        st.info("You have no booked appointments yet.")


def main():
    st.title('Diabetes Risk Prediction & Prevention Hub')

    # Initialize session state
    if 'assessments' not in st.session_state:
        st.session_state['assessments'] = {}
    if 'show_minh_history' not in st.session_state:
        st.session_state['show_minh_history'] = False
    if 'posts' not in st.session_state:
        st.session_state['posts'] = [
            {'title': 'New to the Community!', 'content': "Hi everyone, I'm new here and looking forward to learning more about diabetes prevention."},
            {'title': 'Question about Diet', 'content': "What are some good low-sugar snack options?", 'comments': ["Nuts and seeds are a good choice!", "Greek yogurt with berries.", "Vegetable sticks with hummus."]},
            {'title': 'Exercise Tips', 'content': "Does anyone have advice for staying motivated to exercise regularly?"}
        ] # Initialize posts with some content
    if 'appointments' not in st.session_state: # Initialize appointments list
        st.session_state['appointments'] = []


    with st.sidebar:
        st.header("Navigation")
        # Add "Doctor Appointment" to the menu
        menu = ["Risk Assessment", "Prevention Articles", "Assessment History", "Doctor Appointment", "About", "Discussion Forum"]
        choice = st.selectbox("Go to", menu)
        st.markdown("---")
        st.info("Developed by Healthcare & Coding Enthusiasts with the goal to better Sickness Prevention")

    if choice == "Risk Assessment":
        risk_assessment_page()
    elif choice == "Prevention Articles":
        articles_page()
    elif choice == "Assessment History":
        assessment_history_page()
    elif choice == "Doctor Appointment": # Handle new menu item
        doctor_appointment_page()
    elif choice == "About":
        about_page()
    elif choice == "Discussion Forum":
        discussion_forum_page()

if __name__ == '__main__':
    main()
