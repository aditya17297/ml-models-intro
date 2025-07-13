import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# import data sets
dataset = pd.read_csv('Data.csv')
x = dataset.iloc[:, :-1].values  # all the rows put `:` and all the columns except last then put `:-1` which means all the columns except the index of last column i.e -1
y = dataset.iloc[:, -1].values
print(x)
print(y)


# Taking care of missing data -> replace by average of the column values

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])
print(x)


# Encoding Categorical Data -> split Country column in 3 binary valued columns since we have 3 countries

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
x = np.array(ct.fit_transform(x))
print(x)


# Encoding Categorical Data -> set yes as 1 and no as 0

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)
print(y)

# Splitting the dataset into the training set and Test set

from  sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

print(x_train)
print(x_test)
print(y_train)
print(y_test)

# feature scaling

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train[:, 3:] = sc.fit_transform(x_train[:, 3:])
x_test[:, 3:] = sc.transform(x_test[:, 3:])

print(x_train)
print(x_test)
