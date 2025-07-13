import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

## importing dataset
dataset = pd.read_csv("/Users/adityaagrawal/PycharmProjects/PythonProject/3_ClassificationMLModels/1_LogisticRegression/Social_Network_Ads.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

## splitting test and training data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

## Feature Scaling

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

##training model
from sklearn.svm import SVC
model = SVC(kernel="rbf", random_state=0)
model.fit(X_train, y_train)

## predict

y_pred = model.predict(X_test)
# print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_pred), 1)), 1))

## Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
acs = accuracy_score(y_test, y_pred)
print(cm)
print(acs)

## Applying KFold Cross Validation
from sklearn.model_selection import cross_val_score
accuracyList = cross_val_score(estimator=model, X=X_train, y=y_train, cv=10)

print(f'mean : {accuracyList.mean()}')
print(f'standard deviation : {accuracyList.std()}')
