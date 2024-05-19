import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pickle
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor


# Read the CSV file
df = pd.read_csv("survey_results_public.csv")

# Select relevant columns and rename for clarity
df = df[["Country", "EdLevel", "YearsCodePro", "Employment", "ConvertedComp"]]
df = df.rename({"ConvertedComp": "Salary"}, axis=1)

# Remove rows with null values
df = df.dropna()

# Filter to include only full-time employees
df = df[df["Employment"] == "Employed full-time"]

# Drop the Employment column as it's now redundant
df = df.drop("Employment", axis=1)

# countries with less than 400 respondents into 'Other' category
def clean_countries(count, cutoff):
    categorical_map = {}
    for i in range(len(count)):
        if count.values[i] >= cutoff:
            categorical_map[count.index[i]] = count.index[i]
        else:
            categorical_map[count.index[i]] = 'Other'
    return categorical_map

country_map = clean_countries(df.Country.value_counts(), 400)
df['Country'] = df['Country'].map(country_map)

# Filter salary ranges between 30k and 250k and exclude 'Other' countries
df = df[df["Salary"] <= 250000]
df = df[df["Salary"] >= 30000]
df = df[df['Country'] != 'Other']

# Convert YearsCodePro to numeric
def clean_experience(x):
    if x == 'More than 50 years':
        return 50
    if x == 'Less than 1 year':
        return 0.5
    return float(x)

df['YearsCodePro'] = df['YearsCodePro'].apply(clean_experience)

# Simplify education levels
def clean_education(x):
    if 'Bachelor’s degree' in x:
        return 'Bachelor’s degree'
    if 'Master’s degree' in x:
        return 'Master’s degree'
    if 'Professional degree' in x or 'Other doctoral' in x:
        return 'Post grad'
    return 'Less than a Bachelors'

df['EdLevel'] = df['EdLevel'].apply(clean_education)

# Encode categorical variables
le_education = LabelEncoder()
df['EdLevel'] = le_education.fit_transform(df['EdLevel'])

le_country = LabelEncoder()
df['Country'] = le_country.fit_transform(df['Country'])

# Define features and labels
X = df.drop("Salary", axis=1)
y = df["Salary"]

# Split the data into training and testing sets. Only 20 percent of the data is testing data and rest 80 percent is training data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

'''
# Initialize and train the Linear Regression model
linear = linear_model.LinearRegression()
linear.fit(X_train, y_train)

acc = linear.score(X_test, y_test)
# print(acc)

# Predict on the test set
y_pred = linear.predict(X_test)

# now lets see how much error is in my model
error = np.sqrt(mean_squared_error(y_test, y_pred))
print(error)
'''

#38k error is pretty high. so Linear regression is not good. Lets try other regression models from sklearn
#Decision Tree regressor
decisionTree = DecisionTreeRegressor(random_state=0)
decisionTree.fit(X, y.values)

predictions = decisionTree.predict(X_test)

error = np.sqrt(mean_squared_error(y_test, predictions))
# print(error) #this seems reasonably low = 29.6k


# country, edlevel, yearscode - new array inputted by the user
X = np.array([["United States", 'Bachelor’s degree', 7 ]])

#lets apply the label encoder for the params then predict the salary
X[:, 0] = le_country.transform(X[:,0])
X[:, 1] = le_education.transform(X[:,1])
X = X.astype(float)
# print(X)

# test_pred = decisionTree.predict(X)
# print(test_pred)
#ok it works - predicted salary - [118578.19823789]

#Lets save our model
data = {"model": decisionTree, "le_country": le_country, "le_education": le_education}
with open('salaryPrediction.pickle', 'wb') as f:
    pickle.dump(data, f)

#lets test if it worked.
with open('salaryPrediction.pickle', 'rb') as f:
    data = pickle.load(f)

#accessing the key
loaded = data["model"]
le_country = data["le_country"]
le_education = data["le_education"]

test_pred = loaded.predict(X)
print(test_pred)
#works - [118578.19823789] - same value as before

#works