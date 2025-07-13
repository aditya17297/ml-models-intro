## Predicting if the customer will leave the bank or not based on the data provided

import numpy as np
import pandas as pd
import tensorflow as tf
import keras

##### Data Preprocessing

# Importing dataset
dataset = pd.read_csv('/Users/adityaagrawal/PycharmProjects/PythonProject/8_DeepLearning/1_ArtifitialNeuralNetwork/Churn_Modelling.csv')
X = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values

# Encoding Categorical Data --> Gender
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:, 2])

# Encode categorical DAta --> Country
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

# Split into test and training set
from  sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


##### Building Artificial Neural Network

# initialise Artificial Neural Network
ann = keras.models.Sequential()

# add input and first hidden layer
ann.add(keras.layers.Dense(units=6, activation='relu'))  # rectifier activation function for hidden layers

# add second hidden layer
ann.add(keras.layers.Dense(units=6, activation='relu'))  # rectifier activation function for hidden layers

## add output layer
ann.add(keras.layers.Dense(units=1, activation='sigmoid'))  # sigmoid activation function for binary and softmax for categorical classification


##### Training Artificial Neural Network

# compile Artificial Neural Network
ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) # for binary classification loss = binary_crossentropy otherwise categorical_crossentropy

# training Artificial Neural Network
ann.fit(X_train, y_train, batch_size = 32, epochs=100)


##### Predict test set result
y_pred = ann.predict(X_test)
y_pred = (y_pred > 0.5)
# print (np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))

## Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
acs = accuracy_score(y_test, y_pred)
print(cm)
print(acs)