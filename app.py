
import streamlit as st
import pandas as pd
import joblib
import os
import plotly.express as px

from datetime import datetime

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Rockfall AI | Risk Intelligence",
    page_icon="🪨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 2. DARK THEME AND CSS
# ==========================================

st.markdown("""
<style>

.stApp {
    background-color: #111827;
    color: #F9FAFB;
}

[data-testid="stSidebar"] {
    background-color: #0B1120;
    border-right: 1px solid #374151;
}

[data-testid="stHeader"] {
    background-color: transparent;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
}

h1, h2, h3 {
    color: #F9FAFB !important;
}

p, label {
    color: #D1D5DB;
}

.metric-card {
    background: #1F2937;
    border: 1px solid #374151;
    border-radius: 12px;
    padding: 18px;
}

.metric-label {
    color: #9CA3AF;
    font-size: 13px;
    text-transform: uppercase;
}

.metric-value {
    color: #F59E0B;
    font-size: 28px;
    font-weight: bold;
}

.stButton > button {
    background-color: #F59E0B;
    color: #111827;
    border: none;
    border-radius: 8px;
    font-weight: bold;
    min-height: 45px;
}

.stButton > button:hover {
    background-color: #D97706;
    color: white;
}

hr {
    border-color: #374151;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# 3. LOAD TRAINED MODEL
# ==========================================

@st.cache_resource
def load_model():
    return joblib.load("model/rockfall_model.pkl")


try:
    model = load_model()

except Exception as e:
    st.error(f"Unable to load AI model: {e}")
    st.stop()


HISTORY_FILE = "prediction_history.csv"


# ==========================================
# 4. HELPER FUNCTIONS
# ==========================================

def load_history():

    if os.path.exists(HISTORY_FILE):
        return pd.read_csv(HISTORY_FILE)

    return pd.DataFrame()


def save_prediction(record):

    new_record = pd.DataFrame([record])

    if os.path.exists(HISTORY_FILE):

        old_history = pd.read_csv(HISTORY_FILE)

        updated_history = pd.concat(
            [old_history, new_record],
            ignore_index=True
        )

    else:
        updated_history = new_record

    updated_history.to_csv(
        HISTORY_FILE,
        index=False
    )


def get_recommendation(risk):

    recommendations = {

        "Low": [
            "Continue routine slope monitoring.",
            "Inspect the area periodically.",
            "Maintain regular geological assessments."
        ],

        "Medium": [
            "Increase monitoring frequency.",
            "Inspect cracks and loose rocks.",
            "Consider assessment by geological experts."
        ],

        "High": [
            "Restrict access to potentially hazardous areas.",
            "Notify responsible authorities.",
            "Seek assessment from qualified geological experts."
        ]
    }

    return recommendations.get(risk, [])


def show_metric(label, value):

    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
    </div>
    """, unsafe_allow_html=True)


# ==========================================
# 5. SIDEBAR NAVIGATION
# ==========================================

with st.sidebar:

    st.title("🪨 ROCKFALL AI")

    st.caption("GEOLOGICAL RISK INTELLIGENCE")

    st.divider()

    st.success("AI MODEL CONNECTED")

    page = st.radio(
        "MAIN MENU",
        [
            "Dashboard",
            "Risk Prediction",
            "Analytics",
            "Prediction History",
            "About Project"
        ]
    )

    st.divider()

    st.subheader("MODEL INFORMATION")

    st.write("Algorithm: Random Forest")

    st.write("Decision Trees: 100")

    st.write("Training Records: 4,000")

    st.write("Testing Records: 1,000")

    st.write("Test Accuracy: 85.2%")

    st.divider()

    st.caption("Academic Prototype | 2026")


# ==========================================
# 6. LOAD PREDICTION HISTORY
# ==========================================

history = load_history()


# ==========================================
# 7. DASHBOARD PAGE
# ==========================================

if page == "Dashboard":

    st.title("🪨 Rockfall AI Command Centre")

    st.caption(
        "Environmental monitoring and AI risk intelligence"
    )

    st.divider()

    total = len(history)

    if total > 0:

        high_count = (
            history["Predicted Risk"] == "High"
        ).sum()

        medium_count = (
            history["Predicted Risk"] == "Medium"
        ).sum()

        low_count = (
            history["Predicted Risk"] == "Low"
        ).sum()

    else:

        high_count = 0
        medium_count = 0
        low_count = 0

    # KPI CARDS

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        show_metric("Total Predictions", total)

    with col2:
        show_metric("High Risk", high_count)

    with col3:
        show_metric("Medium Risk", medium_count)

    with col4:
        show_metric("Low Risk", low_count)

    st.write("")

    left, right = st.columns(2)

    # RISK DISTRIBUTION

    with left:

        st.subheader("📊 Risk Distribution")

        if total > 0:

            distribution = (
                history["Predicted Risk"]
                .value_counts()
                .reindex(
                    ["Low", "Medium", "High"],
                    fill_value=0
                )
                .reset_index()
            )

            distribution.columns = [
                "Risk Level",
                "Count"
            ]

            fig = px.pie(
                distribution,
                names="Risk Level",
                values="Count",
                hole=0.6,
                color="Risk Level",
                color_discrete_map={
                    "Low": "#22C55E",
                    "Medium": "#F59E0B",
                    "High": "#EF4444"
                }
            )

            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        else:

            st.info("No predictions available yet.")

    # MODEL INFORMATION

    with right:

        st.subheader("🤖 AI Model Information")

        st.markdown("""
        **Algorithm:** Random Forest Classifier

        **Number of Trees:** 100

        **Training Data:** 4,000 records

        **Testing Data:** 1,000 records

        **Test Accuracy:** 85.2%

        **Output Classes:**
        - Low Risk
        - Medium Risk
        - High Risk
        """)

    # RECENT PREDICTIONS

    st.divider()

    st.subheader("🕒 Recent Predictions")

    if total > 0:

        st.dataframe(
            history.tail(5).iloc[::-1],
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info(
            "Your recent predictions will appear here."
        )


# ==========================================
# 8. RISK PREDICTION PAGE
# ==========================================

elif page == "Risk Prediction":

    st.title("🎯 Rockfall Risk Prediction")

    st.caption(
        "Enter environmental parameters to generate an AI-based risk classification."
    )

    st.divider()

    st.subheader("🌦️ Environmental Parameters")

    st.write(
        "Adjust the values according to the conditions being evaluated."
    )

    # INPUT FORM

    with st.form("prediction_form"):

        col1, col2 = st.columns(2)

        with col1:

            rainfall = st.slider(
                "🌧️ Rainfall (mm)",
                0.0, 100.0, 30.0
            )

            slope_angle = st.slider(
                "⛰️ Slope Angle (degrees)",
                15.0, 85.0, 45.0
            )

            soil_moisture = st.slider(
                "💧 Soil Moisture (%)",
                5.0, 100.0, 40.0
            )

        with col2:

            ground_vibration = st.slider(
                "📳 Ground Vibration",
                0.0, 10.0, 3.0
            )

            rock_strength = st.slider(
                "🪨 Rock Strength",
                0.0, 100.0, 60.0
            )

        submitted = st.form_submit_button(
            "🔍 PREDICT ROCKFALL RISK",
            use_container_width=True
        )

    # RUN PREDICTION

    if submitted:

        input_data = pd.DataFrame({

            "Rainfall": [rainfall],

            "Slope_Angle": [slope_angle],

            "Soil_Moisture": [soil_moisture],

            "Ground_Vibration": [ground_vibration],

            "Rock_Strength": [rock_strength]

        })

        prediction = model.predict(input_data)

        probabilities = model.predict_proba(input_data)

        risk = prediction[0]

        # Store prediction in session

        st.session_state["last_prediction"] = {
            "risk": risk,
            "probabilities": probabilities[0],
            "inputs": input_data
        }

        # Save history

        record = {

            "Date and Time": datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            ),

            "Rainfall": rainfall,

            "Slope Angle": slope_angle,

            "Soil Moisture": soil_moisture,

            "Ground Vibration": ground_vibration,

            "Rock Strength": rock_strength,

            "Predicted Risk": risk

        }

        save_prediction(record)

        st.session_state["last_prediction_time"] = (
            record["Date and Time"]
        )

    # DISPLAY PREDICTION RESULT

    if "last_prediction" in st.session_state:

        result = st.session_state["last_prediction"]

        risk = result["risk"]

        probabilities = result["probabilities"]

        st.divider()

        st.subheader("🤖 AI Prediction Result")

        # NATIVE STREAMLIT RISK DISPLAY
        # No custom HTML here

        if risk == "Low":

            st.success("🟢 LOW ROCKFALL RISK")

        elif risk == "Medium":

            st.warning("🟡 MEDIUM ROCKFALL RISK")

        else:

            st.error("🔴 HIGH ROCKFALL RISK")

        # Probability information

        probability = max(probabilities) * 100

        st.metric(
            "Highest Class Probability",
            f"{probability:.2f}%"
        )

        # PROBABILITY CHART

        st.subheader("📊 Prediction Probabilities")

        probability_data = pd.DataFrame({

            "Risk Level": model.classes_,

            "Probability (%)": probabilities * 100

        })

        fig = px.bar(
            probability_data,
            x="Risk Level",
            y="Probability (%)",
            color="Risk Level",
            text_auto=".1f",
            color_discrete_map={
                "Low": "#22C55E",
                "Medium": "#F59E0B",
                "High": "#EF4444"
            }
        )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            yaxis_range=[0, 100],
            showlegend=False
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # SAFETY RECOMMENDATIONS

        st.divider()

        st.subheader("🛡️ Safety Recommendations")

        recommendations = get_recommendation(risk)

        for item in recommendations:

            st.markdown(f"- {item}")

        st.caption(
            "These are educational suggestions, not validated emergency instructions."
        )

        st.success("Prediction saved successfully!")

    else:

        st.info(
            "Adjust the parameters and click Predict to see the AI result."
        )


# ==========================================
# 9. ANALYTICS PAGE
# ==========================================

elif page == "Analytics":

    st.title("📈 Environmental Risk Analytics")

    st.caption(
        "Explore patterns in previously generated predictions."
    )

    if len(history) > 0:

        st.subheader("🌧️ Rainfall vs Slope Angle")

        fig = px.scatter(
            history,
            x="Rainfall",
            y="Slope Angle",
            color="Predicted Risk",
            size="Ground Vibration",
            hover_data=[
                "Soil Moisture",
                "Rock Strength"
            ],
            color_discrete_map={
                "Low": "#22C55E",
                "Medium": "#F59E0B",
                "High": "#EF4444"
            }
        )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.subheader("📊 Parameter Analysis")

        selected_parameter = st.selectbox(
            "Select environmental parameter",
            [
                "Rainfall",
                "Slope Angle",
                "Soil Moisture",
                "Ground Vibration",
                "Rock Strength"
            ]
        )

        fig2 = px.histogram(
            history,
            x=selected_parameter,
            color="Predicted Risk",
            barmode="overlay",
            opacity=0.75,
            color_discrete_map={
                "Low": "#22C55E",
                "Medium": "#F59E0B",
                "High": "#EF4444"
            }
        )

        fig2.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )

        st.plotly_chart(
            fig2,
            use_container_width=True
        )

    else:

        st.info(
            "Generate predictions first to unlock analytics."
        )


# ==========================================
# 10. PREDICTION HISTORY PAGE
# ==========================================

elif page == "Prediction History":

    st.title("🕒 Prediction History")

    st.caption(
        "Review and download previously generated predictions."
    )

    if len(history) > 0:

        st.metric(
            "Total Predictions",
            len(history)
        )

        risk_filter = st.multiselect(
            "Filter by risk level",
            ["Low", "Medium", "High"],
            default=["Low", "Medium", "High"]
        )

        filtered_history = history[
            history["Predicted Risk"].isin(risk_filter)
        ]

        st.dataframe(
            filtered_history.iloc[::-1],
            use_container_width=True,
            hide_index=True
        )

        csv_data = filtered_history.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "⬇️ Download Prediction History",
            data=csv_data,
            file_name="rockfall_prediction_history.csv",
            mime="text/csv",
            use_container_width=True
        )

    else:

        st.info("No prediction records available yet.")


# ==========================================
# 11. ABOUT PROJECT PAGE
# ==========================================

elif page == "About Project":

    st.title("ℹ️ About Rockfall AI")

    st.subheader(
        "Smart AI-Based Rockfall Prediction System"
    )

    st.write("""
    This academic machine learning prototype classifies
    potential rockfall risk using environmental parameters.

    ### Machine Learning

    - Algorithm: Random Forest Classifier
    - Decision Trees: 100
    - Dataset: 5,000 synthetic observations
    - Training Split: 80%
    - Testing Split: 20%
    - Test Accuracy: 85.2%

    ### Input Parameters

    1. Rainfall
    2. Slope Angle
    3. Soil Moisture
    4. Ground Vibration
    5. Rock Strength

    ### Output Categories

    - Low Risk
    - Medium Risk
    - High Risk

    ### Technology Stack

    - Python
    - NumPy
    - Pandas
    - Scikit-learn
    - Streamlit
    - Plotly
    - Joblib
    """)

    st.warning(
        "This prototype uses synthetic training data. "
        "Its predictions are not validated for real-world "
        "geological hazard assessment."
    )


# ==========================================
# 12. FOOTER
# ==========================================

st.divider()

st.caption(
    "ROCKFALL AI | GEOLOGICAL RISK INTELLIGENCE | 2026"
)
