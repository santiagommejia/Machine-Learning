from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import pandas as pd

# Notes V1:
# Applied Feature Scaling
# Mean squared error: 
# Coefficient of determination:

df = pd.read_csv('NYHouseDataset.csv')
df = df.drop(['BROKERTITLE', 'TYPE', 'ADDRESS', 'STATE', 'MAIN_ADDRESS', 'ADMINISTRATIVE_AREA_LEVEL_2', 'LOCALITY', 'SUBLOCALITY', 'STREET_NAME', 'LONG_NAME', 'FORMATTED_ADDRESS'], axis=1)
X = df.drop(['PRICE'], axis=1) # X = FEATURES
Y = df['PRICE']

# perform feature scaling for the 'PRICE' column on X
# scaler = StandardScaler()
# X = scaler.fit_transform(X)
# # print the max and min values of X for each column
# print(X.max(axis=0))
# print(X.min(axis=0))

# perform mean normalization
Y = (Y - Y.mean()) / Y.std()
X = (X - X.mean()) / X.std()
print(X.max(axis=0))
print(X.min(axis=0))



# Splitting the dataset into the Training set and Test set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)
model = linear_model.LinearRegression()
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)

# ver mejores maneras de mostrar el squared error, graficar y ver como se puede tener menos error
print('Mean squared error: %.2f' % mean_squared_error(Y_test, Y_pred))
print('Coefficient of determination: %.2f' % r2_score(Y_test, Y_pred))
