import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import sqlite3
from datetime import datetime


# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------

st.set_page_config(
    page_title="Disease Symptoms Analysis and Diagnosis Assistance",
    layout="wide"
)

st.title("🏥 Disease Symptoms Analysis and Diagnosis Assistance")

# -------------------------------------------------
# LOAD MODEL
# -------------------------------------------------

model = joblib.load("disease_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# -------------------------------------------------
# DATABASE CONNECTION
# -------------------------------------------------

conn = sqlite3.connect(
    "patient_records.db",
    check_same_thread=False
)

cursor = conn.cursor()

cursor.execute("SELECT COUNT(*) FROM patient_history")
total_patients = cursor.fetchone()[0]

colA, colB = st.columns([4,1])

with colA:
    st.markdown("### Clinical Decision Support System")

with colB:
    st.metric(
        "Total Records",
        total_patients
    )

# -------------------------------------------------
# CREATE TABLE IF NOT EXISTS
# -------------------------------------------------

cursor.execute("""
CREATE TABLE IF NOT EXISTS patient_history(

patient_id TEXT,

patient_name TEXT,

age INTEGER,

gender TEXT,

symptoms TEXT,

predicted_disease TEXT,

confidence REAL,

temperature REAL,

heart_rate INTEGER,

spo2 INTEGER,

risk_level TEXT,

recommendation TEXT,

visit_date TEXT

)
""")

conn.commit()

# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------

st.sidebar.header("Patient Information")

patient_id = st.sidebar.text_input(
    "Patient ID"
)

patient_name = st.sidebar.text_input(
    "Patient Name"
)

age = st.sidebar.number_input(
    "Age",
    min_value=1,
    max_value=120,
    value=30
)

gender = st.sidebar.selectbox(
    "Gender",
    ["Male","Female","Other"]
)

temperature = st.sidebar.number_input(
    "Temperature (°F)",
    value=98.6
)

heart_rate = st.sidebar.number_input(
    "Heart Rate (BPM)",
    value=80
)

spo2 = st.sidebar.number_input(
    "SpO₂ (%)",
    value=98
)

symptoms = st.text_area(
    "Enter Symptoms",
    height=150,
    placeholder="Example : fever cough headache"
)
# -------------------------------------------------
# SEARCH PREVIOUS PATIENT HISTORY
# -------------------------------------------------

st.sidebar.markdown("---")

search_history = st.sidebar.button("🔍 Search Patient History")

if search_history:

    if patient_id.strip() == "":
        st.error("Please enter Patient ID.")

    else:

        cursor.execute("""
        SELECT
            patient_name,
            visit_date,
            predicted_disease,
            confidence,
            risk_level
        FROM patient_history
        WHERE patient_id = ?
        ORDER BY visit_date DESC
        """, (patient_id,))

        records = cursor.fetchall()

        if len(records) == 0:
            
            st.warning("No previous patient records found.")
        
        else:
            
            latest_name = records[0][0]
            
            st.sidebar.success(f"✅ Patient Found : {latest_name}")
            
            st.success("Patient History Found")
            st.info(f"""
            ### 👤 Patient Details
            **Patient ID:** {patient_id}
            
            **Patient Name:** {latest_name}

            **Last Visit:** {records[0][1]}  
            
            **Last Diagnosed Disease:** {records[0][2]}  
            
            **Last Confidence:** {records[0][3]}%  
            
            **Last Risk Level:** {records[0][4]}
            """)
            
            
            st.subheader("📋 Previous Patient Records")
            
            history = []
            
            for row in records:
                
                history.append({
                    
                    "Patient Name": row[0],
                    
                    "Visit Date": row[1],
                    
                    "Disease": row[2],
                    
                    "Confidence (%)": row[3],
                    
                    "Risk": row[4]
                
                })
                
            history_df = pd.DataFrame(history)
            
            st.dataframe(
                
                history_df,
                use_container_width=True,
                hide_index=True
            )
            
            csv = history_df.to_csv(index=False).encode("utf-8")
            
            st.download_button(
                
                label="⬇ Download Patient History",
                
                data=csv,
                
                file_name=f"{patient_id}_history.csv",
                
                mime="text/csv"
            
            )
# -------------------------------------------------
# ANALYZE PATIENT
# -------------------------------------------------

st.markdown("---")

analyze = st.button("🩺 Analyze Patient")

if analyze:

    # Validation
    if patient_id.strip() == "":
        st.error("Please enter Patient ID.")

    elif patient_name.strip() == "":
        st.error("Please enter Patient Name.")

    elif symptoms.strip() == "":
        st.error("Please enter Patient Symptoms.")

    else:

        # Convert symptoms into TF-IDF vector
        symptom_vector = vectorizer.transform([symptoms])

        # Disease Prediction
        prediction = model.predict(symptom_vector)[0]

        # Probability
        probability = model.predict_proba(symptom_vector)[0]

        confidence = round(max(probability) * 100, 2)
        
        # Top 5 Differential Diagnosis

        disease_prob = list(zip(model.classes_, probability))

        disease_prob = sorted(
            
            disease_prob,
            
            key=lambda x: x[1],
            
            reverse=True
        
        )
        
        top5 = disease_prob[:5]


        # -------------------------
        # Risk Assessment
        # -------------------------

        if spo2 < 90:

            emergency = "CRITICAL"

        elif temperature > 102:

            emergency = "HIGH RISK"

        elif heart_rate > 120:

            emergency = "MODERATE"

        else:

            emergency = "NORMAL"
        # -------------------------------------------------
        # DOCTOR RECOMMENDATION
        # -------------------------------------------------

        if emergency == "CRITICAL":
            recommendation = "Immediate ICU Admission"

        elif emergency == "HIGH RISK":
            recommendation = "Urgent Medical Investigation Required"

        elif emergency == "MODERATE":
            recommendation = "Doctor Consultation Recommended"

        else:
            recommendation = "Routine Medical Consultation"

        # -------------------------------------------------
        # DISPLAY RESULTS
        # -------------------------------------------------

        st.markdown("---")

        st.subheader("🩺 Prediction Result")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Predicted Disease",
                prediction
            )

        with col2:
            st.metric(
                "Confidence",
                f"{confidence}%"
            )

        with col3:
            st.metric(
                "Risk Level",
                emergency
            )

        st.success(f"Doctor Recommendation : {recommendation}")
        # -------------------------------------------------
        # SAVE PATIENT RECORD
        # -------------------------------------------------

        cursor.execute("""
        INSERT INTO patient_history
        (
            patient_id,
            patient_name,
            age,
            gender,
            symptoms,
            predicted_disease,
            confidence,
            temperature,
            heart_rate,
            spo2,
            risk_level,
            recommendation,
            visit_date
        )

        VALUES
        (?,?,?,?,?,?,?,?,?,?,?,?,?)

        """,

        (

            patient_id,
            patient_name,
            age,
            gender,
            symptoms,
            prediction,
            confidence,
            temperature,
            heart_rate,
            spo2,
            emergency,
            recommendation,
            datetime.now().strftime("%d-%m-%Y %H:%M")

        )

        )

        conn.commit()

        st.success("✅ Patient Record Saved Successfully!")
        
        st.markdown("---")
        
        st.subheader("📋 Patient Summary")
        
        summary = pd.DataFrame({
            
            "Field":[
                
            "Patient ID",
                
            "Patient Name",
                
            "Age",
                
            "Gender",
                
            "Symptoms",
                
            "Predicted Disease",
                
            "Confidence",
                
            "Risk Level",
                
            "Recommendation",
                
            "Visit Date"
            
            ],
            
            "Value":[
                
            patient_id,
                
            patient_name,
                
            age,
                
            gender,
                
            symptoms,
                
            prediction,
                
            f"{confidence}%",
                
            emergency,
                
            recommendation,
                
            datetime.now().strftime("%d-%m-%Y %H:%M")
            
            ]
        })
        
        st.table(summary)

        st.markdown("---")
        
        st.subheader("📊 Prediction Confidence")
        
        chart = pd.DataFrame({
            
            "Prediction": ["Confidence"],
            
            "Percentage": [confidence]
        })
        
        fig = px.bar(
            
            chart,
            
            x="Prediction",
            
            y="Percentage",
            
            text="Percentage"
        )
        
        fig.update_layout(height=350)
        
        st.plotly_chart(
            
            fig,
            
            use_container_width=True
        )

        st.markdown("---")
        
        st.subheader("🩺 Top 5 Differential Diagnosis")
        
        top5_df = pd.DataFrame(
            
            top5,
            
            columns=[
                
                "Disease",
                
                "Probability"
            ]
        )
        
        top5_df["Probability"] = top5_df["Probability"] * 100
        
        fig = px.bar(
            
            top5_df,
            
            x="Probability",

            y="Disease",
            
            orientation="h",
            
            text="Probability"
        
        )
        
        fig.update_layout(
            
            height=400,
            
            yaxis={'categoryorder':'total ascending'}
        
        )
        
        st.plotly_chart(
            
            fig,
            
            use_container_width=True
        )

