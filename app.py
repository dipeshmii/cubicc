import streamlit as st
import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# ==========================================
# 1. CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Smart Container Loading",
    page_icon="🚢",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .metric-box { padding: 20px; border-radius: 10px; text-align: center; margin-bottom: 20px; }
    .good { background-color: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
    .avg { background-color: #fff3cd; color: #856404; border: 1px solid #ffeeba; }
    .poor { background-color: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. MODEL LOADING & SELF-HEALING LOGIC
# ==========================================
MODEL_FILE = 'cube_utilisation_model.pkl'
DATA_FILE = 'improved_dataset.csv'

def train_model_internal():
    """Trains the model inside the app if the .pkl file is broken or missing."""
    if not os.path.exists(DATA_FILE):
        return None

    df = pd.read_csv(DATA_FILE)
    X = df.drop(columns=['cube_utilisation_pct'])
    y = df['cube_utilisation_pct']

    categorical_features = ['pallet_pattern', 'packing_orientation']
    numerical_features = [col for col in X.columns if col not in categorical_features]

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    model_pipeline.fit(X, y)
    joblib.dump(model_pipeline, MODEL_FILE, compress=3)
    return model_pipeline

@st.cache_resource
def load_model():
    try:
        # Try loading the existing file
        return joblib.load(MODEL_FILE)
    except (FileNotFoundError, AttributeError, Exception) as e:
        # If loading fails (Version mismatch or missing file), RETRAIN
        st.warning(f"⚠️ Model version mismatch detected ({e}). Retraining model automatically...")
        return train_model_internal()

# Load the model (or retrain it)
model = load_model()

# ==========================================
# 3. UI & INPUTS
# ==========================================
st.sidebar.title("🚢 Container Config")
container_type = st.sidebar.radio("Container Type:", ["20ft Standard", "40ft Standard", "Custom"])

if container_type == "20ft Standard":
    c_len, c_wid, c_hgt = 5.9, 2.35, 2.39
    default_w_limit = 28000
    st.sidebar.info("✅ Standard 20ft dimensions applied.")
elif container_type == "40ft Standard":
    c_len, c_wid, c_hgt = 12.0, 2.35, 2.39
    default_w_limit = 29000
    st.sidebar.info("✅ Standard 40ft dimensions applied.")
else:
    c_len = st.sidebar.number_input("Length (m)", 2.0, 20.0, 6.0)
    c_wid = st.sidebar.number_input("Width (m)", 1.0, 4.0, 2.35)
    c_hgt = st.sidebar.number_input("Height (m)", 1.0, 4.0, 2.39)
    default_w_limit = 28000

w_limit = st.sidebar.number_input("Max Weight Limit (kg)", 1000, 50000, default_w_limit)
c_vol = c_len * c_wid * c_hgt
st.sidebar.markdown(f"**Total Capacity:** `{c_vol:.2f} m³`")

# Main Inputs
st.title("📦 Container Utilisation Predictor")
col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Cargo Details")
    small_boxes = st.number_input("Small Boxes Count", 0, 2000, 150)
    large_boxes = st.number_input("Large Boxes Count", 0, 500, 40)
    total_boxes = small_boxes + large_boxes
    total_vol = st.number_input("Total Cargo Volume (m³)", 1.0, 80.0, 25.0)
    curr_weight = st.number_input("Total Cargo Weight (kg)", 100, 50000, 12000)
    irregular = st.number_input("Irregular Parts", 0, 50, 0)

with col2:
    st.subheader("2. Packing Strategy")
    pattern = st.selectbox("Pallet Pattern", ["pinwheel", "brick", "stacked", "mixed"])
    orient = st.selectbox("Packing Orientation", ["LWH", "WHL", "mixed"])
    st.write("---")
    predict_btn = st.button("🚀 Predict Efficiency", type="primary", use_container_width=True)

# ==========================================
# 4. PREDICTION
# ==========================================
if predict_btn:
    if model is None:
        st.error("🚨 Model could not be loaded or trained. Please ensure 'improved_dataset.csv' is in the GitHub repository.")
    else:
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

        try:
            prediction = model.predict(input_df)[0]
            
            if prediction >= 85:
                status, css, msg = "Excellent", "good", "Great job! Max efficiency."
            elif prediction >= 65:
                status, css, msg = "Average", "avg", "Acceptable, but could be better."
            else:
                status, css, msg = "Poor", "poor", "High wasted space detected."

            st.markdown(f"""
                <div class="metric-box {css}">
                    <h2 style="margin:0;">{prediction:.2f}%</h2>
                    <p style="font-size:18px; margin:0;"><b>{status}</b></p>
                </div>
            """, unsafe_allow_html=True)
            st.info(msg)
            st.progress(min(prediction/100, 1.0))
            
        except Exception as e:
            st.error(f"Prediction Error: {e}")
