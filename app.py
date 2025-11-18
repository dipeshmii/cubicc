import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Page Configuration
st.set_page_config(
    page_title="Cube Utilisation Predictor",
    page_icon="📦",
    layout="centered"
)

# 1. Load the Model
@st.cache_resource
def load_model():
    try:
        return joblib.load('cube_utilisation_model.pkl')
    except FileNotFoundError:
        st.error("Model file not found! Please ensure 'cube_utilisation_model.pkl' is in the same directory.")
        return None

model = load_model()

# 2. App Header
st.title("📦 Cube Utilisation Prediction")
st.markdown("Predict the efficiency of your truck/container load plan based on physical constraints and packing strategies.")
st.write("---")

# 3. User Inputs (Sidebar for configuration, Main for dimensions)
st.subheader("🚛 Load Configuration")

col1, col2, col3 = st.columns(3)

with col1:
    truck_len = st.number_input("Truck Length (m)", min_value=2.0, max_value=20.0, value=7.5)
    total_boxes = st.number_input("Total Box Count", min_value=1, value=160)
    weight_limit = st.number_input("Max Weight Limit (kg)", min_value=100.0, value=2000.0)

with col2:
    truck_width = st.number_input("Truck Width (m)", min_value=1.0, max_value=5.0, value=2.4)
    total_box_vol = st.number_input("Total Box Volume (m³)", min_value=0.1, value=35.0)
    total_weight = st.number_input("Actual Load Weight (kg)", min_value=10.0, value=1800.0)

with col3:
    truck_height = st.number_input("Truck Height (m)", min_value=1.0, max_value=5.0, value=2.6)
    irregular_count = st.number_input("Irregular Parts", min_value=0, value=5)
    
st.write("---")
st.subheader("📦 Packing Strategy")

col4, col5 = st.columns(2)

with col4:
    # Breakdown of boxes
    small_boxes = st.number_input("Small Box Count", min_value=0, value=100)
    large_boxes = st.number_input("Large Box Count", min_value=0, value=60)

with col5:
    # Categorical inputs matching the training data
    pallet_pattern = st.selectbox("Pallet Pattern", ["pinwheel", "mixed", "stacked", "brick"])
    packing_orientation = st.selectbox("Packing Orientation", ["LWH", "mixed", "WHL"])

# 4. Automatic Calculations (Hidden Logic)
# The model expects 'truck_volume_m3' and 'avg_box_volume_m3'
truck_volume = truck_len * truck_width * truck_height
avg_box_vol = total_box_vol / total_boxes if total_boxes > 0 else 0

# 5. Prediction Logic
if st.button("🚀 Predict Utilisation", type="primary"):
    if model is not None:
        # Create a DataFrame matching the training features EXACTLY
        input_data = pd.DataFrame({
            'truck_length_m': [truck_len],
            'truck_width_m': [truck_width],
            'truck_height_m': [truck_height],
            'truck_volume_m3': [truck_volume],
            'total_boxes': [total_boxes],
            'total_box_volume_m3': [total_box_vol],
            'avg_box_volume_m3': [avg_box_vol],
            'small_box_count': [small_boxes],
            'large_box_count': [large_boxes],
            'irregular_parts_count': [irregular_count],
            'pallet_pattern': [pallet_pattern],
            'packing_orientation': [packing_orientation],
            'weight_limit_kg': [weight_limit],
            'total_weight_kg': [total_weight]
        })

        # Get Prediction
        prediction = model.predict(input_data)[0]
        
        # Display Result
        st.write("---")
        st.subheader("📊 Results")
        
        # Logic for Color and Suggestions
        if prediction > 85:
            color = "green"
            status = "Excellent"
            msg = "Great job! This load plan is highly efficient."
        elif prediction >= 60:
            color = "orange" # Streamlit uses 'gold' or 'orange' usually implies warning
            status = "Acceptable"
            msg = "The load is okay, but there is room for optimization."
        else:
            color = "red"
            status = "Poor"
            msg = "Low utilisation detected. Consider re-planning."

        # Big Metric Display
        st.markdown(f"""
        <div style="text-align: center; padding: 20px; background-color: #f0f2f6; border-radius: 10px; border: 2px solid {color};">
            <h2 style="color: {color}; margin:0;">{prediction:.2f}%</h2>
            <p style="font-size: 1.2em; margin:0;"><b>{status}</b></p>
        </div>
        """, unsafe_allow_html=True)
        
        st.info(msg)

        # Suggestions Section
        if prediction < 85:
            st.subheader("💡 Optimization Suggestions")
            if packing_orientation == "mixed":
                st.markdown("- **Try Uniform Orientation:** Switching to specific 'LWH' might stabilize stacking.")
            if irregular_count > 10:
                st.markdown("- **Irregular Items:** High number of irregular parts (tyres/mirrors) significantly reduces cube utilisation. Try to separate them.")
            if pallet_pattern == "stacked":
                st.markdown("- **Pattern Change:** 'Pinwheel' patterns often utilize corners better than simple 'Stacked' patterns.")
            st.markdown("- **Box Mixing:** Ensure small boxes are used to fill gaps created by large boxes.")

    else:
        st.error("Model not loaded. Cannot predict.")

# Footer
st.markdown("---")
st.caption("Logistics Optimization Model • Powered by Random Forest & Streamlit")
