from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

# Notes V1:
# removing the outliers from the dataset, values that are over 10 million, excludes 157 entries from the set of 4801 inputs
# Mean squared error: 1452208.85
# Coefficient of determination: 0.30

df = pd.read_csv('NYHouseDataset.csv')
df = df.drop(['BROKERTITLE', 'TYPE', 'ADDRESS', 'STATE', 'MAIN_ADDRESS', 'ADMINISTRATIVE_AREA_LEVEL_2', 'LOCALITY', 'SUBLOCALITY', 'STREET_NAME', 'LONG_NAME', 'FORMATTED_ADDRESS'], axis=1)
X = df.drop(['PRICE'], axis=1) # X = FEATURES
Y = df['PRICE'] 

Y = Y / 1000 # Convert to thousands of dollars
indexes = Y[Y > 10000].index # Remove outliers worth more than 10 million
print(len(indexes))
Y = Y.drop(indexes)
X = X.drop(indexes)

# Splitting the dataset into the Training set and Test set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)
model = linear_model.LinearRegression()
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)

print('Mean squared error: %.2f' % mean_squared_error(Y_test, Y_pred))
print('Coefficient of determination: %.2f' % r2_score(Y_test, Y_pred))
