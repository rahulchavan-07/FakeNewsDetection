# fake_news_detection.py

import numpy as np
import pandas as pd
import re
import nltk
import joblib
import warnings
warnings.filterwarnings('ignore')

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

nltk.download('stopwords')

# Load Dataset
news_dataset = pd.read_csv('fake_news_dataset.csv')
news_dataset = news_dataset.fillna('')
news_dataset['label'] = news_dataset['label'].map({'real': 0, 'fake': 1})
news_dataset['content'] = news_dataset['author'] + ' ' + news_dataset['title'] + ' ' + news_dataset['category'] + ' ' + news_dataset['source']

# Stemming
stemmer = PorterStemmer()
def stemming(content):
    content = re.sub('[^a-zA-Z]', ' ', content)
    content = content.lower().split()
    content = [stemmer.stem(word) for word in content if word not in stopwords.words('english')]
    return ' '.join(content)

news_dataset['content'] = news_dataset['content'].apply(stemming)

# Feature and Label Separation
X = news_dataset['content'].values
Y = news_dataset['label'].values

# Vectorization
vectorizer = TfidfVectorizer()
vectorizer.fit(X)
X = vectorizer.transform(X)

# Train-Test Split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=42)

# Model Definitions
models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(n_estimators=100),
    "Linear SVM": LinearSVC()
}

# Training & Evaluation
best_model = None
best_accuracy = 0
best_model_name = ""

for name, model in models.items():
    model.fit(X_train, Y_train)
    pred = model.predict(X_test)
    accuracy = accuracy_score(Y_test, pred)
    print(f"{name} Test Accuracy: {accuracy:.4f}")
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model
        best_model_name = name

print(f"\n✅ Best Model: {best_model_name} with accuracy: {best_accuracy:.4f}")

# Save vectorizer and best model using joblib
joblib.dump((vectorizer, best_model), 'best_model.joblib')

# Predict one example
X_new = X_test[0]
prediction = best_model.predict(X_new)

print("\nPrediction for 1st test sample:")
print("Real News" if prediction[0] == 0 else "Fake News")
print("Actual Label:", "Real News" if Y_test[0] == 0 else "Fake News")
