import streamlit as st

# Set the page configuration
st.set_page_config(
    page_title="SWE Salary Predictor",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="auto"
)

from predict import viewPage
from data import viewData

# apply custom CSS styling
with open("style.css") as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

curr = st.sidebar.selectbox("View Data or View Prediction", ("View Prediction", "View Data"))

if curr == "View Prediction":
    viewPage()
else:
    viewData()