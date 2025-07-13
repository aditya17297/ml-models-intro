import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

## Importing the data set
dataSet = pd.read_csv("/Users/adityaagrawal/PycharmProjects/PythonProject/2_RegressionMLModels/3_PolynomialLinearRegression/Position_Salaries.csv")
X = dataSet.iloc[:, 1:-1].values
y = dataSet.iloc[:, -1].values

## Training the Linear Regression Model on whole Data set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X, y)

## Training the Polynomial Regression Modal on whole Dataset

# Convert the feature in matrix of polynomial like x, x^2, x^3, x^4 .... x^n and then fit the new data in linear regression
from sklearn.preprocessing import PolynomialFeatures
poly_regr = PolynomialFeatures(degree=4)
X_poly = poly_regr.fit_transform(X)
lin_regr = LinearRegression()
lin_regr.fit(X_poly, y)

## Visualize the Linear Regression results
plt.scatter(X, y, color='red')
plt.plot(X, regressor.predict(X), color='blue')
plt.title('Linear Regression Model')
plt.xlabel('Position Level')
plt.ylabel('Salaries')
plt.show()


## Visualize the Polynomial Linear Regression results
plt.scatter(X, y, color='red')
plt.plot(X, lin_regr.predict(X_poly), color='blue')
plt.title('Polynomial Linear Regression Model')
plt.xlabel('Position Level')
plt.ylabel('Salaries')
plt.show()

# Visualising the Polynomial Regression results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, lin_regr.predict(poly_regr.fit_transform(X_grid)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Predicting a new result with Linear Regression
print(regressor.predict([[6.5]]))

# Predicting a new result with Polynomial Regression
print(lin_regr.predict(poly_regr.fit_transform([[6.5]])))



