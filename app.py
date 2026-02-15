import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load trained model and scaler
model = joblib.load("model/best_model.pkl")
scaler = joblib.load("model/scaler.pkl")
label_encoder = joblib.load("model/label_encoder.pkl")

st.title("Dry Bean Classification App")

st.write("Enter bean measurements to predict the bean type.")

# Input fields
area = st.number_input("Area", min_value=0.0)
perimeter = st.number_input("Perimeter", min_value=0.0)
major_axis = st.number_input("Major Axis Length", min_value=0.0)
minor_axis = st.number_input("Minor Axis Length", min_value=0.0)
aspect_ratio = st.number_input("Aspect Ratio", min_value=0.0)
eccentricity = st.number_input("Eccentricity", min_value=0.0)
convex_area = st.number_input("Convex Area", min_value=0.0)
equiv_diameter = st.number_input("Equivalent Diameter", min_value=0.0)
extent = st.number_input("Extent", min_value=0.0)
solidity = st.number_input("Solidity", min_value=0.0)
roundness = st.number_input("Roundness", min_value=0.0)
compactness = st.number_input("Compactness", min_value=0.0)
shape_factor1 = st.number_input("Shape Factor 1", min_value=0.0)
shape_factor2 = st.number_input("Shape Factor 2", min_value=0.0)
shape_factor3 = st.number_input("Shape Factor 3", min_value=0.0)
shape_factor4 = st.number_input("Shape Factor 4", min_value=0.0)

if st.button("Predict Bean Type"):
    features = np.array([[area, perimeter, major_axis, minor_axis,
                          aspect_ratio, eccentricity, convex_area,
                          equiv_diameter, extent, solidity, roundness,
                          compactness, shape_factor1, shape_factor2,
                          shape_factor3, shape_factor4]])
    
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)
    bean_type = label_encoder.inverse_transform(prediction)

    st.success(f"Predicted Bean Type: {bean_type[0]}")
