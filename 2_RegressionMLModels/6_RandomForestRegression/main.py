import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

## Importing the data set
dataSet = pd.read_csv("/Users/adityaagrawal/PycharmProjects/PythonProject/2_RegressionMLModels/5_DecisionTreeRegression/Position_Salaries.csv")
X = dataSet.iloc[:, 1:-1].values
y = dataSet.iloc[:, -1].values

## Training Random Forrest Model on data set
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=10, random_state=0)## n_estimators indicate number if trees
regressor.fit(X, y)

## Predicting the value
y_pred = regressor.predict([[6.5]])
print(y_pred)

## Visualizing the Random Forest Regression Result
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Random Forest Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
