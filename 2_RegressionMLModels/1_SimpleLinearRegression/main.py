#Data Prediction using Simple Linear Regression ML Model

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Importing dataset from file
dataset = pd.read_csv("/Users/adityaagrawal/PycharmProjects/PythonProject/2_RegressionMLModels/1_SimpleLinearRegression/Salary_Data.csv")
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Split data set into training and testing datasets
from  sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

# Training SLR on training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# Predicting test data result
y_pred = regressor.predict(x_test)

# visualising the training data set
plt.scatter(x_train, y_train, color = 'red')
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.title('Salaries vs Experience (Training Set)')
plt.xlabel('YOE')
plt.ylabel('Salary')
plt.show()


# visualising the testing data set
plt.scatter(x_test, y_test, color = 'red')
plt.plot(x_train, regressor.predict(x_train), color = 'blue') ## no need to change the parameters since linear line would be the same
plt.title('Salaries vs Experience (Training Set)')
plt.xlabel('YOE')
plt.ylabel('Salary')
plt.show()
