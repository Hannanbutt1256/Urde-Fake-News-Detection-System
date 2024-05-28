import pandas as pd
import re
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle

# Load the CSV data
data = pd.read_csv(
    r'C:\Users\hanna\OneDrive\Documents\FYP\Model Training\Combined .csv')

# Load stopwords from JSON file
with open('stopwords.json', 'r', encoding='utf-8') as file:
    stopwords_data = json.load(file)

    # Check if the loaded JSON is a dictionary or a list
    if isinstance(stopwords_data, dict):
        stopwords = stopwords_data['stopwords']
    elif isinstance(stopwords_data, list):
        stopwords = stopwords_data
    else:
        raise ValueError("Invalid stopwords file format.")

# Function to preprocess text


def preprocess_text(text, stopwords):
    # Remove punctuation and lowercase the text
    text = re.sub(r'\W', ' ', text)
    text = text.lower()

    # Remove stopwords
    text = ' '.join([word for word in text.split() if word not in stopwords])

    return text


# Apply preprocessing to the News Items column
data['processed_news'] = data['News Items'].apply(
    lambda x: preprocess_text(str(x), stopwords))

# Extract features using TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data['processed_news'])
# Convert labels to binary (1 for TRUE, 0 for FAKE)
y = data['Label'].apply(lambda x: 1 if x == 'TRUE' else 0)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Initialize the models
logistic_regression = LogisticRegression()
naive_bayes = MultinomialNB()
svm = SVC()

# Train the models
logistic_regression.fit(X_train, y_train)
naive_bayes.fit(X_train, y_train)
svm.fit(X_train, y_train)

# Ensemble Model
ensemble_model = VotingClassifier(estimators=[
    ('lr', logistic_regression),
    ('nb', naive_bayes),
    ('svm', svm)
], voting='hard')

# Train the ensemble model
ensemble_model.fit(X_train, y_train)


# Save the trained model
with open('ensemble_model.pkl', 'wb') as model_file:
    pickle.dump(ensemble_model, model_file)

# Save the vectorizer
with open('tfidf_vectorizer.pkl', 'wb') as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)


# Function to predict new news


def predict_news(news, model, vectorizer, stopwords):
    # Preprocess the input news
    processed_news = preprocess_text(news, stopwords)
    # Transform the news using the trained TF-IDF vectorizer
    news_vector = vectorizer.transform([processed_news])
    # Predict using the trained model
    prediction = model.predict(news_vector)
    # Convert the prediction back to label
    return 'TRUE' if prediction[0] == 1 else 'FAKE'


# Example usage
# Replace with actual Urdu news text
new_news = "سوشل میڈیا پر وائرل مبینہ طور پر خیبرپختونخوا (KP) کے وزیر اعلیٰ علی امین خان گنڈا پور کی اپنی مشیر مشال یوسفزئی کو ایک بالکل نئی لگژری کار کی چابیاں تحفے میں دیتے ہوئے تصویر نے ایک نئی بحث کو جنم دیا ہے۔"
prediction = predict_news(new_news, ensemble_model, vectorizer, stopwords)
print(f"The news is predicted to be: {prediction}")

# Evaluate the models (previous evaluation part of the script)
lr_predictions = logistic_regression.predict(X_test)
nb_predictions = naive_bayes.predict(X_test)
svm_predictions = svm.predict(X_test)
ensemble_predictions = ensemble_model.predict(X_test)

print("Logistic Regression Accuracy: ", accuracy_score(y_test, lr_predictions))
print("Naive Bayes Accuracy: ", accuracy_score(y_test, nb_predictions))
print("SVM Accuracy: ", accuracy_score(y_test, svm_predictions))
print("Ensemble Accuracy: ", accuracy_score(y_test, ensemble_predictions))

print("\nLogistic Regression Classification Report")
print(classification_report(y_test, lr_predictions))

print("\nNaive Bayes Classification Report")
print(classification_report(y_test, nb_predictions))

print("\nSVM Classification Report")
print(classification_report(y_test, svm_predictions))

print("\nEnsemble Classification Report")
print(classification_report(y_test, ensemble_predictions))
