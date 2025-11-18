import streamlit as st
import pandas as pd
import joblib

# ==========================================
# 1. CONFIGURATION & SETUP
# ==========================================
st.set_page_config(
    page_title="Smart Container Loading",
    page_icon="🚢",
    layout="wide"
)

# Custom CSS for metrics
st.markdown("""
    <style>
    .metric-box {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 20px;
    }
    .good { background-color: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
    .avg { background-color: #fff3cd; color: #856404; border: 1px solid #ffeeba; }
    .poor { background-color: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. LOAD MODEL
# ==========================================
@st.cache_resource
def load_model():
    try:
        return joblib.load('cube_utilisation_model.pkl')
    except FileNotFoundError:
        return None

model = load_model()

# ==========================================
# 3. SIDEBAR: CONTAINER CONFIGURATION
# ==========================================
st.sidebar.title("🚢 Container Config")
st.sidebar.markdown("Select standard shipping container sizes or enter custom dimensions.")

container_type = st.sidebar.radio(
    "Container Type:",
    ["20ft Standard", "40ft Standard", "Custom"]
)

# Auto-fill dimensions based on selection (Standard Shipping Specs)
if container_type == "20ft Standard":
    # Internal dims: ~5.9m x 2.35m x 2.39m
    c_len, c_wid, c_hgt = 5.9, 2.35, 2.39
    default_w_limit = 28000
    st.sidebar.info("✅ Standard 20ft dimensions applied.")
    
elif container_type == "40ft Standard":
    # Internal dims: ~12.0m x 2.35m x 2.39m
    c_len, c_wid, c_hgt = 12.0, 2.35, 2.39
    default_w_limit = 29000
    st.sidebar.info("✅ Standard 40ft dimensions applied.")

else:
    # Custom Manual Input
    c_len = st.sidebar.number_input("Length (m)", 2.0, 20.0, 6.0)
    c_wid = st.sidebar.number_input("Width (m)", 1.0, 4.0, 2.35)
    c_hgt = st.sidebar.number_input("Height (m)", 1.0, 4.0, 2.39)
    default_w_limit = 28000

# Weight Limit Input
w_limit = st.sidebar.number_input("Max Weight Limit (kg)", 1000, 50000, default_w_limit)

# Calculate Container Volume automatically
c_vol = c_len * c_wid * c_hgt
st.sidebar.markdown(f"**Total Capacity:** `{c_vol:.2f} m³`")

# ==========================================
# 4. MAIN PAGE: CARGO & STRATEGY
# ==========================================
st.title("📦 Container Utilisation Predictor")
st.markdown("Enter your cargo details and packing strategy to predict **Space Efficiency**.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Cargo Details")
    
    # Box Counts
    c1_sub, c2_sub = st.columns(2)
    with c1_sub:
        small_boxes = st.number_input("Small Boxes Count", 0, 2000, 150, help="Small boxes are good for filling gaps.")
    with c2_sub:
        large_boxes = st.number_input("Large Boxes Count", 0, 500, 40, help="Main cargo bulk.")
    
    total_boxes = small_boxes + large_boxes
    st.caption(f"Total Box Count: {total_boxes}")
    
    # Volumes & Weights
    total_vol = st.number_input("Total Cargo Volume (m³)", 1.0, 80.0, 25.0)
    curr_weight = st.number_input("Total Cargo Weight (kg)", 100, 50000, 12000)
    
    # Irregular Parts (The Cube Killer)
    irregular = st.number_input("Irregular Parts (Tyres/Mirrors)", 0, 50, 0, 
                                help="Non-rectangular items drastically reduce packing efficiency.")

with col2:
    st.subheader("2. Packing Strategy")
    
    # Categorical Inputs
    pattern = st.selectbox(
        "Pallet Pattern", 
        ["pinwheel", "brick", "stacked", "mixed"],
        help="Pinwheel/Brick patterns interlock boxes for better stability and density."
    )
    
    orient = st.selectbox(
        "Packing Orientation", 
        ["LWH", "WHL", "mixed"],
        help="LWH = Upright, WHL = On Side. 'Mixed' usually creates voids."
    )

    st.write("---")
    predict_btn = st.button("🚀 Predict Efficiency", type="primary", use_container_width=True)

# ==========================================
# 5. PREDICTION LOGIC
# ==========================================
if predict_btn:
    if model is None:
        st.error("🚨 Model not found! Please ensure 'cube_utilisation_model.pkl' is in the same folder.")
    else:
        # Create DataFrame with EXACT columns used in training
        input_df = pd.DataFrame({
            'container_length_m': [c_len],
            'container_width_m': [c_wid],
            'container_height_m': [c_hgt],
            'container_volume_m3': [c_vol],
            'total_boxes': [total_boxes],
            'small_box_count': [small_boxes],
            'large_box_count': [large_boxes],
            'total_box_volume_m3': [total_vol],
            'irregular_parts_count': [irregular],
            'pallet_pattern': [pattern],
            'packing_orientation': [orient],
            'weight_limit_kg': [w_limit],
            'total_weight_kg': [curr_weight]
        })

        # Predict
        try:
            prediction = model.predict(input_df)[0]
            
            # Logic for Display
            if prediction >= 85:
                status = "Excellent Efficiency"
                css_class = "good"
                msg = "Great job! This configuration maximizes container space."
            elif prediction >= 65:
                status = "Average Efficiency"
                css_class = "avg"
                msg = "Acceptable, but try adjusting orientation or reducing irregular parts."
            else:
                status = "Poor Efficiency"
                css_class = "poor"
                msg = "Warning: High wasted space detected. Likely due to irregular parts or poor stacking pattern."

            # Display Result
            st.markdown("---")
            st.markdown(f"""
                <div class="metric-box {css_class}">
                    <h2 style="margin:0;">{prediction:.2f}%</h2>
                    <p style="font-size:18px; margin:0;"><b>{status}</b></p>
                </div>
            """, unsafe_allow_html=True)
            
            st.info(msg)
            
            # Progress Bar
            st.progress(min(prediction/100, 1.0))
            
            # Detailed Data
            with st.expander("See Calculation Details"):
                st.write(f"Container Volume: {c_vol:.2f} m³")
                st.write(f"Cargo Volume: {total_vol:.2f} m³")
                st.write(f"Theoretical Max Fill: {(total_vol/c_vol)*100:.2f}%")
                st.write(f"AI Predicted Fill: {prediction:.2f}% (Adjusted for gaps & patterns)")

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
