import numpy as np
import joblib
import streamlit as st
import pandas as pd
import datetime
import altair as alt

# --- Load the Model ---
try:
    model_path = 'trained_model.joblib'
    model = joblib.load(model_path)
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

def save_assessment(user_id, assessment_data):
    st.session_state.setdefault('assessments', {})
    st.session_state['assessments'].setdefault(user_id, []).append(assessment_data)
    st.toast("Assessment saved!", icon="💾")

def load_assessment_history(user_id):
    return st.session_state.get('assessments', {}).get(user_id, [])

# --- Risk Assessment Page ---
def risk_assessment_page():
    st.header("Diabetes Risk Assessment")
    st.subheader("Enter Your Health Information")

    if 'We' not in st.session_state:
        st.session_state['We'] = DEFAULT_WEIGHT
    if 'He' not in st.session_state:
        st.session_state['He'] = DEFAULT_HEIGHT

    col1, col2 = st.columns(2)
    age = col1.number_input('Age (years)', min_value=0, max_value=120, value=st.session_state.get('Age', DEFAULT_AGE))
    pregnancies = col1.number_input('Number of Pregnancies', min_value=0, max_value=20, value=st.session_state.get('Pregnancies', DEFAULT_PREGNANCIES))
    glucose = col1.number_input('Glucose Level (mg/dL)', min_value=0, max_value=300, value=st.session_state.get('Glucose', DEFAULT_GLUCOSE))
    blood_pressure = col1.number_input('Blood Pressure (mmHg)', min_value=0, max_value=200, value=st.session_state.get('BloodPressure', DEFAULT_BLOODPRESSURE))
    skin_thickness = col2.number_input('Skin Thickness (mm)', min_value=0, max_value=100, value=st.session_state.get('SkinThickness', DEFAULT_SKINTHICKNESS))
    insulin = col2.number_input('Insulin Level (mu U/mL)', min_value=0, max_value=850, value=st.session_state.get('Insulin', DEFAULT_INSULIN))
    weight = col2.number_input('Weight (kg)', min_value=20, max_value=300, value=st.session_state['We'], on_change=lambda: st.session_state.__setitem__('We', weight))
    height = col2.number_input('Height (meters)', min_value=0.5, max_value=3.0, value=st.session_state['He'], step=0.01, on_change=lambda: st.session_state.__setitem__('He', height))
    diabetes_pedigree_function = col1.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=3.0, value=st.session_state.get('DiabetesPedigreeFunction', DEFAULT_DIABETESPEDIGREEFUNCTION), step=0.001)

    bmi = calculate_bmi(weight, height)
    st.metric("Calculated BMI", f"{bmi:.2f}")
    st.session_state['BMI'] = bmi

    if st.button("Assess Risk"):
        if loaded_model:
            input_data = [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]
            risk_probability = diabetes_prediction_proba(input_data, loaded_model)
            risk_level = get_risk_level(risk_probability)

            st.subheader(f"Estimated Diabetes Risk: {risk_level} ({risk_probability * 100:.2f}%)")
            display_risk_interpretation(risk_level)

            # Since login is removed, we can use a fixed user ID or None
            save_assessment("guest", assessment_data)
        else:
            st.error("Model not loaded. Cannot assess risk.")

def assessment_history_page():
    st.header("Assessment History")
    # Fabricated data for user 'minh' - 12 months
    current_year = 2024
    minh_history_data = [
        {'timestamp': datetime.datetime(current_year, 6, 1), 'age': 34, 'pregnancies': 0, 'glucose': 115, 'blood_pressure': 82, 'skin_thickness': 26, 'insulin': 60, 'bmi': 25.0, 'diabetes_pedigree_function': 0.58, 'risk_probability': 0.35, 'risk_level': 'Moderate'},
        {'timestamp': datetime.datetime(current_year, 7, 1), 'age': 34, 'pregnancies': 0, 'glucose': 108, 'blood_pressure': 80, 'skin_thickness': 24, 'insulin': 55, 'bmi': 24.8, 'diabetes_pedigree_function': 0.55, 'risk_probability': 0.32, 'risk_level': 'Moderate'},
        {'timestamp': datetime.datetime(current_year, 8, 1), 'age': 34, 'pregnancies': 0, 'glucose': 102, 'blood_pressure': 78, 'skin_thickness': 22, 'insulin': 50, 'bmi': 24.5, 'diabetes_pedigree_function': 0.50, 'risk_probability': 0.28, 'risk_level': 'Moderate'},
        {'timestamp': datetime.datetime(current_year, 9, 1), 'age': 35, 'pregnancies': 0, 'glucose': 98, 'blood_pressure': 76, 'skin_thickness': 20, 'insulin': 45, 'bmi': 24.2, 'diabetes_pedigree_function': 0.48, 'risk_probability': 0.25, 'risk_level': 'Low'},
        {'timestamp': datetime.datetime(current_year, 10, 1), 'age': 35, 'pregnancies': 0, 'glucose': 95, 'blood_pressure': 80, 'skin_thickness': 22, 'insulin': 40, 'bmi': 24.5, 'diabetes_pedigree_function': 0.45, 'risk_probability': 0.15, 'risk_level': 'Low'},
        {'timestamp': datetime.datetime(current_year, 11, 1), 'age': 35, 'pregnancies': 0, 'glucose': 105, 'blood_pressure': 83, 'skin_thickness': 25, 'insulin': 50, 'bmi': 24.7, 'diabetes_pedigree_function': 0.51, 'risk_probability': 0.29, 'risk_level': 'Moderate'},
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

    st.subheader("Risk Probability Over Time (12 Months)")
    chart_risk = alt.Chart(df_minh).mark_line(point=True).encode(
        x=alt.X('timestamp:T', title='Month'),
        y=alt.Y('risk_probability:Q', title='Risk Probability'),
        tooltip=['timestamp:T', 'risk_probability:Q', 'risk_level:N']
    ).properties(
        title='Risk Probability Trend'
    ).interactive()
    st.altair_chart(chart_risk, use_container_width=True)

    st.subheader("Glucose Levels Over Time (12 Months)")
    chart_glucose = alt.Chart(df_minh).mark_line(point=True).encode(
        x=alt.X('timestamp:T', title='Month'),
        y=alt.Y('glucose:Q', title='Glucose (mg/dL)'),
        tooltip=['timestamp:T', 'glucose:Q']
    ).properties(
        title='Glucose Level Trend'
    ).interactive()
    st.altair_chart(chart_glucose, use_container_width=True)

    st.subheader("Blood Pressure Over Time (12 Months)")
    chart_bp = alt.Chart(df_minh).mark_line(point=True).encode(
        x=alt.X('timestamp:T', title='Month'),
        y=alt.Y('blood_pressure:Q', title='Blood Pressure (mmHg)'),
        tooltip=['timestamp:T', 'blood_pressure:Q']
    ).properties(
        title='Blood Pressure Trend'
    ).interactive()
    st.altair_chart(chart_bp, use_container_width=True)

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

def main():
    st.title('Diabetes Risk Prediction & Prevention Hub')

    # Initialize session state
    if 'assessments' not in st.session_state:
        st.session_state['assessments'] = {}
    if 'show_minh_history' not in st.session_state:
        st.session_state['show_minh_history'] = False

    with st.sidebar:
        st.header("Navigation")
        menu = ["Risk Assessment", "Prevention Articles", "Assessment History", "About"]
        choice = st.selectbox("Go to", menu)
        st.markdown("---")
        st.info("Developed by Healthcare & Coding Enthusiasts with the goal to better Sickness Prevention")

    if choice == "Risk Assessment":
        risk_assessment_page()
    elif choice == "Prevention Articles":
        articles_page()
    elif choice == "Assessment History":
        assessment_history_page()
    elif choice == "About":
        about_page()

    # Add fabricated data for 'minh' if it doesn't exist
    if 'minh' not in st.session_state['assessments']:
        st.session_state['assessments']['minh'] = [
            {'timestamp': '2025-05-01T10:00:00', 'age': 35, 'pregnancies': 0, 'glucose': 95, 'blood_pressure': 80, 'skin_thickness': 22, 'insulin': 40, 'bmi': 24.5, 'diabetes_pedigree_function': 0.45, 'risk_probability': 0.15, 'risk_level': 'Low'},
            {'timestamp': '2025-04-25T14:30:00', 'age': 35, 'pregnancies': 0, 'glucose': 110, 'blood_pressure': 85, 'skin_thickness': 25, 'insulin': 55, 'bmi': 24.8, 'diabetes_pedigree_function': 0.52, 'risk_probability': 0.28, 'risk_level': 'Moderate'},
            {'timestamp': '2025-04-18T09:15:00', 'age': 34, 'pregnancies': 0, 'glucose': 102, 'blood_pressure': 78, 'skin_thickness': 20, 'insulin': 48, 'bmi': 24.0, 'diabetes_pedigree_function': 0.40, 'risk_probability': 0.18, 'risk_level': 'Low'},
            {'timestamp': '2025-04-10T16:40:00', 'age': 34, 'pregnancies': 0, 'glucose': 115, 'blood_pressure': 82, 'skin_thickness': 26, 'insulin': 60, 'bmi': 25.0, 'diabetes_pedigree_function': 0.58, 'risk_probability': 0.35, 'risk_level': 'Moderate'},
        ]

if __name__ == '__main__':
    main()
