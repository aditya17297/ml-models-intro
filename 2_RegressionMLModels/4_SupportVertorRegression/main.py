import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

## Importing the data set
dataSet = pd.read_csv("/Users/adityaagrawal/PycharmProjects/PythonProject/2_RegressionMLModels/4_SupportVertorRegression/Position_Salaries.csv")
X = dataSet.iloc[:, 1:-1].values
y = dataSet.iloc[:, -1].values

## Feature Scaling

y = y.reshape(len(y), 1)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)


## Training SVR model on data set
from sklearn.svm import SVR
regressor = SVR(kernel='rbf')
regressor.fit(X, y)

## Predict a new Result
y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform([[6.5]])).reshape(-1, 1))
print(y_pred)

## Visualising the SVR Result
plt.scatter(sc_X.inverse_transform(X), sc_y.inverse_transform(y), color='red')
plt.plot(sc_X.inverse_transform(X), sc_y.inverse_transform(regressor.predict(X).reshape(-1, 1)), color='blue')
plt.title('Polynomial Linear Regression Model')
plt.xlabel('Position Level')
plt.ylabel('Salaries')
plt.show()


# Visualising the SVR results (for higher resolution and smoother curve)
X_grid = np.arange(min(sc_X.inverse_transform(X)), max(sc_X.inverse_transform(X)), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(sc_X.inverse_transform(X), sc_y.inverse_transform(y), color = 'red')
plt.plot(X_grid, sc_y.inverse_transform(regressor.predict(sc_X.transform(X_grid)).reshape(-1, 1)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()