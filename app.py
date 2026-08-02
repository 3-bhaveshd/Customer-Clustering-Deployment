import streamlit as st
from pipeline import InferencePipeline

st.set_page_config(
    page_title="Customer Segmentation App",
    page_icon="🛍️",
    layout="centered"
)

st.title("🛍️ Customer Segmentation Predictor")
st.write("Enter customer attributes below to determine their cluster category.")

@st.cache_resource
def get_inference_pipeline():
    return InferencePipeline()

try:
    pipeline = get_inference_pipeline()
    
    with st.form("customer_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            gender = st.selectbox("Gender", options=["Male", "Female"])
            age = st.number_input("Age", min_value=18, max_value=100, value=30, step=1)
            
        with col2:
            annual_income = st.number_input("Annual Income ($k)", min_value=10, max_value=300, value=50, step=1)
            spending_score = st.slider("Spending Score (1-100)", min_value=1, max_value=100, value=50)
            
        submit_button = st.form_submit_button("Predict Customer Segment")

    if submit_button:
        predicted_category = pipeline.predict_segment(
            gender=gender,
            age=age,
            annual_income=annual_income,
            spending_score=spending_score
        )
        
        st.success("Target Category Identified!")
        st.metric(label="Predicted Segment", value=predicted_category)

except Exception as e:
    st.error(f"Error loading model artifacts: {e}")
    st.info("Make sure you have run `python train.py` to generate 'preprocessor.joblib' and 'kmeans_model.joblib' in your root directory.")