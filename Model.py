#Importing Libraries
import pandas as pd
import numpy as np

#Loading Dataset
df = pd.read_csv('balanced_reviews.csv')

#Checking shape/Columns/Head/Sample of Data
df.shape
df.columns
df.head()
df.head(10)
df.sample(10)

#Checking first review of Data and Overall Column Counts
df['reviewText'][0]
df['overall'].value_counts()

#Handaling null values
df.isnull().any(axis = 0)
df.dropna(inplace = True)

#leaving the reviews with rating 3 and collect reviews with
#rating 1, 2, 4 and 5 onyl
df['overall'] == 3
df [df['overall'] == 3]
df = df [df['overall'] != 3]

#creating a label
#based on the values in overall column
np.where(df['overall'] > 3, 1, 0)
df['Positivity'] = np.where(df['overall'] > 3, 1, 0)

#NLP
#reviewText - feature - df['reviewText']
#Positivity - label - df['Positivity']

#version #01
#countVectorizer
#Importing train_test_split from sklearn
from sklearn.model_selection import train_test_split

features_train, features_test, labels_train, labels_test = train_test_split(df['reviewText'], df['Positivity'], random_state= 42)

#Importing feature_extractor - CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer

vect = CountVectorizer().fit(features_train)

len(vect.get_feature_names())

vect.get_feature_names()[10000:10010]

features_train_vectorized = vect.transform(features_train)

#features_train_vectorized.toarray()
#create the classifier

#Importing Logistic Regression Model for prection of version 1 i.e., for countVectorizer
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()

model.fit(features_train_vectorized, labels_train)

predictions = model.predict(vect.transform(features_test))

#Importing confusion matrix to check the score of model
from sklearn.metrics import confusion_matrix
confusion_matrix(labels_test, predictions)

#Importing roc_auc_score to check the score of model
from sklearn.metrics import roc_auc_score
roc_auc_score(labels_test, predictions)

#version 02
#tf-idf 
#term frequency inverse document frequency
#Importing train_test_split
from sklearn.model_selection import train_test_split

features_train, features_test, labels_train, labels_test = train_test_split(df['reviewText'], df['Positivity'], random_state = 42 )

#Importing TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

vect = TfidfVectorizer(min_df = 5).fit(features_train)
len(vect.get_feature_names())

vect.get_feature_names()[10000:10010]

features_train_vectorized = vect.transform(features_train)


#features_train_vectorized.toarray()
#create the classifier
#Importing Logistic Regression model for prediction of Version 2 i.e., for TfidfVectorizer
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

model.fit(features_train_vectorized, labels_train)

predictions = model.predict(vect.transform(features_test))

#Importing confusion matrix to check the score of model
from sklearn.metrics import confusion_matrix

confusion_matrix(labels_test, predictions)

#importing roc_auc_score predictior
from sklearn.metrics import roc_auc_score
roc_auc_score(labels_test, predictions)

#get the better prediction for version 2 as compare to version 1
#so continue with version 2
#tf-idf 
#term frequency inverse document frequency

#Importing train_test_split
from sklearn.model_selection import train_test_split

features_train, features_test, labels_train, labels_test = train_test_split(df['reviewText'], df['Positivity'], random_state = 42 )

#Importing TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

vect = TfidfVectorizer(min_df = 5).fit(features_train)
features_train_vectorized = vect.transform(features_train)

#model building
#Import Logistic Regretion model for prediction of TfidfVectorizer
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(features_train_vectorized, labels_train)
predictions = model.predict(vect.transform(features_test))

#Importing Confusion_matrix for score calculation
from sklearn.metrics import confusion_matrix
confusion_matrix(labels_test, predictions)

#Importing roc_auc_score for score calculation
from sklearn.metrics import roc_auc_score
roc_auc_score(labels_test, predictions)

#save - pickle format for Dashboard building 
#Importing library
import pickle

file  = open("pickle_model.pkl","wb")
pickle.dump(model, file)

#pickle the vocabulary
pickle.dump(vect.vocabulary_, open('features.pkl', 'wb'))
