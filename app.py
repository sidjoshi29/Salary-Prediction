import streamlit as st
from predict import viewPage
from data import viewData

curr = st.sidebar.selectbox("View Data or View Prediction", ("View Prediction", "View Data"))

if curr == "View Prediction":
    viewPage()
else:
    viewData()