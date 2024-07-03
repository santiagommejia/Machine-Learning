from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

BostonHousing = pd.read_csv('BostonHousing.csv')
Y = BostonHousing.medv
X = BostonHousing.drop(['medv'], axis=1)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
print(X.shape, Y.shape)
model = linear_model.LinearRegression()
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)

print('Mean squared error: %.2f' % mean_squared_error(Y_test, Y_pred))
print('Coefficient of determination: %.2f' % r2_score(Y_test, Y_pred))

print (X_train.shape, Y_train.shape)
# Plotting the trained linear regression model
plt.figure(figsize=(10, 6))

# Plotting testing data
testRows = Y_test.shape[0]
testRowsArray = [i for i in range(testRows)]
plt.scatter(testRowsArray, Y_test, color='green', label='Testing data')

# Plotting regression line
predRows = Y_pred.shape[0]
predRowsArray = [i for i in range(predRows)]
plt.scatter(predRowsArray, Y_pred, color='red', label='Prediction Line')

# Adding labels and title
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression')
plt.legend()

# Show plot
plt.show()
