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

    country_list = (
        "United States",
        "India",
        "United Kingdom",
        "Germany",
        "Canada",
        "Brazil",
        "France",
        "Spain",
        "Australia",
        "Netherlands",
        "Poland",
        "Italy",
        "Russian Federation",
        "Sweden",
    )

    education_level = (
        "Less than a Bachelors",
        "Bachelor’s degree",
        "Master’s degree",
        "Post grad",
    )

    country_selected = st.selectbox("Choose Country", country_list)

    education_selected = st.selectbox("Choose Education Level", education_level)

    experience_selected = st.text_input("Years of Experience")

    btn = st.button("Predict the Salary using the above data")
    if(btn) :
        X = np.array([[country_selected, education_selected, experience_selected]])
        X[:, 0] = country.transform(X[:, 0])
        X[:, 1] = education.transform(X[:, 1])
        X = X.astype(float)

        salary = loaded.predict(X)
        st.write(f"The Estimated SALARY is ${salary[0]}.")



