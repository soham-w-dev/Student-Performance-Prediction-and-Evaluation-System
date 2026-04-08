import streamlit as st
import numpy as np
import pickle
import tensorflow as tf

# Load assets from the models folder
model = tf.keras.models.load_model('models/student_model.h5', compile=False)
with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

st.title("Student Performance Evaluation System")

# Inputs
hours = st.number_input("Study Hours", 0, 24, 6)
scores = st.number_input("Previous Scores", 0, 100, 70)
extra = st.selectbox("Extracurricular Activities", ["Yes", "No"])
sleep = st.number_input("Sleep Hours", 0, 12, 8)
papers = st.number_input("Sample Papers Practiced", 0, 10, 5)

if st.button("Predict Performance"):
    # Convert "Yes/No" to 1/0
    extra_val = 1 if extra == "Yes" else 0
    
    # Prepare and scale input
    input_data = np.array([[hours, scores, extra_val, sleep, papers]])
    input_scaled = scaler.transform(input_data)
    
    # Predict with ANN
    prediction = model.predict(input_scaled)
    
    st.subheader("🤖 Neural Network Prediction")
    st.metric("Predicted Performance Index", f"{prediction[0][0]:.2f}")
    
    # NEW: Predict with Fuzzy Logic
    import sys
    sys.path.append('src')
    from fuzzy_logic import get_fuzzy_recommendation
    
    fuzzy_score, fuzzy_text = get_fuzzy_recommendation(hours, sleep, scores)
    
    st.subheader("🧠 Fuzzy Logic Analysis")
    st.info(f"**Expert Recommendation:** {fuzzy_text}")
    st.progress(int(fuzzy_score) / 100)
    st.caption(f"Fuzzy Routine Health Score: {fuzzy_score:.1f}/100")