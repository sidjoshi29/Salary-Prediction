import streamlit as st
import pickle
import numpy as np

#load the file first then access the different keys
def loadFile():
    with open('salaryPrediction.pickle', 'rb') as f:
        data = pickle.load(f)
    return data

data = loadFile()

#loading the keys
loaded = data["model"]
country = data["le_country"]
education = data["le_education"]

def viewPage():
    st.title("SDE Salary Predictor")
    st.write("""### Input Information to predict the salary""")

