## Predict power output from a power plant based on the parameters like temp, humidity, etc.

import numpy as np
import pandas as pd
import tensorflow as tf
import keras

##### Data Preprocessing

# Importing dataset
dataset = pd.read_excel('/Users/adityaagrawal/PycharmProjects/PythonProject/8_DeepLearning/1_ArtifitialNeuralNetwork/Folds5x2_pp.xlsx')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Split into test and training set
from  sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

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
ann.add(keras.layers.Dense(units=1))


##### Training Artificial Neural Network

# compile Artificial Neural Network
ann.compile(optimizer='adam', loss='mean_squared_error') # for binary classification loss = binary_crossentropy otherwise categorical_crossentropy

# training Artificial Neural Network
ann.fit(X_train, y_train, batch_size = 32, epochs=100)


##### Predict test set result
y_pred = ann.predict(X_test)
np.set_printoptions(precision=2)
print (np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))
#
# ## Confusion Matrix
# from sklearn.metrics import confusion_matrix, accuracy_score
# cm = confusion_matrix(y_test, y_pred)
# acs = accuracy_score(y_test, y_pred)
# print(cm)
# print(acs)