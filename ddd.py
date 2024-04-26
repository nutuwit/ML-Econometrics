import pandas as pd
import statsmodels.api as sm
import numpy as np
import seaborn as sns

df = pd.read_csv('HistoricalData_1714146539898.csv')
df = df.drop([0, 1])
df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
df = df.sort_values(by='Date', ascending=True)

# Define the dependent variable
y = df['Close/Last']

# Define the explanatory variables
X = df[['Volume', 'Open', 'High', 'Low']]

# Add a constant to the explanatory variables
X = sm.add_constant(X)

X = X.apply(pd.to_numeric, errors='coerce')
y = y.apply(pd.to_numeric, errors='coerce')

# Create an OLS model
model = sm.OLS(y, X)

# Fit the model to the data
results = model.fit()

# Print the summary of the model
print(results.summary())