from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Notes vChatGpt:
# Setting a polinomial curve to the model
# Mean squared error: 967666.72
# Coefficient of determination: 0.54 

# Load and preprocess data
df = pd.read_csv('NYHouseDataset.csv')
df = df.drop(['BROKERTITLE', 'TYPE', 'ADDRESS', 'STATE', 'MAIN_ADDRESS', 'ADMINISTRATIVE_AREA_LEVEL_2', 'LOCALITY', 'SUBLOCALITY', 'STREET_NAME', 'LONG_NAME', 'FORMATTED_ADDRESS'], axis=1)
X = df.drop(['PRICE'], axis=1)
Y = df['PRICE']

# Convert to thousands of dollars
Y = Y / 1000

# Remove outliers
indexes = Y[Y > 10000].index
Y = Y.drop(indexes)
X = X.drop(indexes)

# Splitting the dataset into the Training set and Test set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# Define a pipeline with StandardScaler, PolynomialFeatures, and LinearRegression
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('poly', PolynomialFeatures(degree=3)),
    ('regressor', LinearRegression())
])

# Fit the model
pipeline.fit(X_train, Y_train)

# Make predictions
Y_pred = pipeline.predict(X_test)

# Evaluate the model
print('Mean squared error: %.2f' % mean_squared_error(Y_test, Y_pred))
print('Coefficient of determination: %.2f' % r2_score(Y_test, Y_pred))

# Plot predictions vs actual values
plt.scatter(Y_test, Y_pred, alpha=0.5)
plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'r--', lw=2)
plt.xlabel("Actual Prices (in thousands)")
plt.ylabel("Predicted Prices (in thousands)")
plt.title("Actual vs Predicted Prices")
plt.show()
