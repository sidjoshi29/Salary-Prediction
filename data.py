import streamlit as st
import pandas as pd
import plotly.express as px

#cleaning the data like how we did in the prediction file
def clean_countries(count, threshold):
    categorical_map = {}
    for i in range(len(count)):
        if count.values[i] >= threshold:
            categorical_map[count.index[i]] = count.index[i]
        else:
            categorical_map[count.index[i]] = 'Other'
    return categorical_map


def clean_experience(x):
    if x == 'More than 50 years':
        return 50
    if x == 'Less than 1 year':
        return 0.5
    return float(x)


def clean_education(x):
    if 'Bachelor’s degree' in x:
        return 'Bachelor’s degree'
    if 'Master’s degree' in x:
        return 'Master’s degree'
    if 'Professional degree' in x or 'Other doctoral' in x:
        return 'Post grad'
    return 'Less than a Bachelors'


#improve the performance
@st.cache_resource
def load_data():
    df = pd.read_csv("survey_results_public.csv")
    df = df[["Country", "EdLevel", "YearsCodePro", "Employment", "ConvertedComp"]]
    df = df[df["ConvertedComp"].notnull()]
    df = df.dropna()
    df = df[df["Employment"] == "Employed full-time"]
    df = df.drop("Employment", axis=1)

    country_map = clean_countries(df.Country.value_counts(), 400)
    df["Country"] = df["Country"].map(country_map)
    df = df[df["ConvertedComp"] <= 500000]
    df = df[df["ConvertedComp"] >= 6500]
    df = df[df["Country"] != "Other"]

    df["YearsCodePro"] = df["YearsCodePro"].apply(clean_experience)
    df["EdLevel"] = df["EdLevel"].apply(clean_education)
    df = df.rename({"ConvertedComp": "Salary"}, axis=1)
    return df

df = load_data()

def viewData():
    st.title("View the data of SWE Salaries")

    data = df["Country"].value_counts()

    st.write("#### Number of Data Entries from Different Countries")
    st.table(data)


    st.write(
        """
    #### Mean Salary Based On Experience
    """
    )

    data = df.groupby(["YearsCodePro"])["Salary"].mean().sort_values(ascending=True)
    st.line_chart(data)

    st.write(
        """
    #### Mean Salary Based On Country
    """
    )

    data = df.groupby(["Country"])["Salary"].mean().sort_values(ascending=True)
    st.bar_chart(data)


    country_mean_salary_map = df.groupby("Country", as_index=False)["Salary"].mean()

    fig2 = px.choropleth(country_mean_salary_map,
                         locations="Country",
                         locationmode="country names",
                         color="Salary",
                         hover_name="Country",
                         color_continuous_scale=px.colors.sequential.Plasma,
                         labels={"Salary": "Mean Salary"})

    fig2.update_geos(projection_type="natural earth")
    fig2.update_layout(geo=dict(showframe=False, showcoastlines=True))
    st.plotly_chart(fig2)