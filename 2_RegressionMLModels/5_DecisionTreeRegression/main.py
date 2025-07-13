import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

## Importing the data set
dataSet = pd.read_csv("/Users/adityaagrawal/PycharmProjects/PythonProject/2_RegressionMLModels/5_DecisionTreeRegression/Position_Salaries.csv")
X = dataSet.iloc[:, 1:-1].values
y = dataSet.iloc[:, -1].values

## Training Decision Tree Regression Model on the dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X, y)

## Predicting a new result
y_pred = regressor.predict([[6.5]])
print(y_pred)

## Visualizing the Decision Tree Regression Result
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Decision Tree Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
