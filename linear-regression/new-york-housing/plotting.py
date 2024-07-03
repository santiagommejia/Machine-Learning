# # Plotting the trained linear regression model
# plt.figure(figsize=(10, 6))

# # Plotting testing data
# testRows = Y_test.shape[0]
# testRowsArray = [i for i in range(testRows)]
# plt.scatter(testRowsArray, Y_test, color='green', label='Testing data')

# # Plotting regression line
# predRows = Y_pred.shape[0]
# predRowsArray = [i for i in range(predRows)]
# plt.scatter(predRowsArray, Y_pred, color='red', label='Prediction Line')

# # Adding labels and title
# plt.xlabel('Sample Number')
# plt.ylabel("House Values (in thousands of dollars)")
# plt.title('Linear Regression')
# plt.legend()

# # Show plot
# plt.show()

