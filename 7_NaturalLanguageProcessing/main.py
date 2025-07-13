import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


## import dataset
dataset = pd.read_csv('/Users/adityaagrawal/PycharmProjects/PythonProject/7_NaturalLanguageProcessing/Restaurant_Reviews.tsv', delimiter='\t', quoting=3)
dataset_size = dataset.values[:,0].size

## Cleaning dataset
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer ## for simplifying words, for example liked to like since both means same

def clean_sentence(uncleaned_sentence):
    all_stop_words = stopwords.words('english')
    all_stop_words.remove('not')
    ps = PorterStemmer()
    cleaned_sentence = re.sub(pattern='[^a-zA-z]', repl=' ', string=uncleaned_sentence)
    cleaned_sentence = cleaned_sentence.lower()
    cleaned_sentence = cleaned_sentence.split()
    cleaned_sentence = [ps.stem(word) for word in cleaned_sentence if not word in set(all_stop_words)]
    cleaned_sentence = ' '.join(cleaned_sentence)
    return cleaned_sentence

cleaned_dataset = []
for i in range(0, dataset_size):
    review = clean_sentence(dataset['Review'][i])
    cleaned_dataset.append(review)

print(cleaned_dataset)

## Create Bag of words
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=1500)
X = cv.fit_transform(cleaned_dataset).toarray()
y = dataset.iloc[:,-1].values

## Split data in test and training set

from  sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

## Training on NAive Byes model
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(X_train, y_train)

## predict test data result
y_pred = model.predict(X_test)

## Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
acs = accuracy_score(y_test, y_pred)
print(cm)
print(acs)
