# import re
# import string
# import joblib
# from nltk.tokenize import word_tokenize
# from nltk.corpus import stopwords
# from sklearn.feature_extraction.text import TfidfVectorizer
#
# # Load the trained model from file
# model = joblib.load('hate_speech_model.joblib')
#
# # Preprocess the input text by cleaning and tokenizing it
# def preprocess_text(text):
#     # Remove URLs, user mentions, and special characters
#     text = re.sub(r'http\S+', '', text)
#     text = re.sub(r'@[A-Za-z0-9_]+', '', text)
#     text = re.sub(r'[^\w\s]', '', text)
#     # Convert to lowercase
#     text = text.lower()
#     # Tokenize the text into words
#     tokens = word_tokenize(text)
#     # Remove stop words
#     stop_words = set(stopwords.words('english'))
#     tokens = [t for t in tokens if not t in stop_words]
#     # Return the cleaned tokens as a string
#     return ' '.join(tokens)
#
# # Define a function that takes a string as input and predicts whether it is hate speech or not
# def predict_hate_speech(text):
#     # Preprocess the input text
#     preprocessed_text = preprocess_text(text)
#     # Predict the label using the trained model
#     y_pred = model.predict([text])
#     print(y_pred)
#     # Return a string indicating whether the input text is hate speech or not
#     if y_pred == 1:
#         return 'This is hate speech.'
#     else:
#         return 'This is not hate speech.'
#
# # Test the function with some example strings
# print(predict_hate_speech('I hate you!'))
# print(predict_hate_speech('The sky is blue.'))


import pandas as pd
import numpy as np
import re

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import joblib

# Load the data
data = pd.read_csv('train.csv')

# Define a function to preprocess the input string
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Remove usernames
    text = re.sub(r'@\S+', '', text)
    # Remove hashtags
    text = re.sub(r'#\S+', '', text)
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove leading/trailing whitespace
    text = text.strip()
    return text

# Preprocess the data
data['text'] = data['text'].apply(preprocess_text)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size=0.2, random_state=42)

# Define the classifiers
lr = LogisticRegression(random_state=42)
nb = MultinomialNB()
rf = RandomForestClassifier(random_state=42)

# Define the TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer()

# Define the voting classifier
voting_clf = VotingClassifier(
    estimators=[('lr', lr), ('nb', nb), ('rf', rf)],
    voting='soft')

# Define the pipeline
pipeline = Pipeline([
    ('tfidf', tfidf_vectorizer),
    ('voting', voting_clf)
])

# Fit the pipeline on the training data
pipeline.fit(X_train, y_train)

# Predict on the test data
y_pred = pipeline.predict(X_test)

# Calculate the accuracy score
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)

# Define a function to predict the label for a given string
def predict_hate_speech(text):
    text = preprocess_text(text)
    label = pipeline.predict([text])[0]
    return label

# Test the function on a sample string
text = "I hate people like you"
label = predict_hate_speech(text)
print('Text:', text)
print('Label:', label)

text = "I love you"
label = predict_hate_speech(text)
print('Text:', text)
print('Label:', label)