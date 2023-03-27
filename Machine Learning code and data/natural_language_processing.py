# Natural Language Processing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset

dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter ='\t', quoting = 3)
# Cleaning the texts
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import  stopwords
from nltk.stem.porter import PorterStemmer
corpus = []   #containing cleaned words
for i in range (0, 1000):
    review = re.sub('[^a-zA-z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')
    review = [ps.stem(word)for word in review if not word in set() ]
    review = ' '.join(review)
    corpus.append(review)
print(corpus)
# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=1200)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, -1].values
len_1 = len(X[0])
print(len_1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Training the Naive Bayes model on the Training set
from sklearn.linear_model import LogisticRegression
prediction = LogisticRegression()
prediction.fit(X_train, y_train)

# Predicting the Test set results
y_pred = prediction.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
f1 = f1_score(y_test, y_pred)
print(f1)
acc = accuracy_score(y_test, y_pred)
print(acc)
matrix = confusion_matrix(y_test,y_pred)
print(matrix)
#   