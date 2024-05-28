from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import re
import json
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the saved model and vectorizer
with open('ensemble_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Load stopwords
with open('stopwords.json', 'r', encoding='utf-8') as file:
    stopwords_data = json.load(file)
    stopwords = stopwords_data if isinstance(
        stopwords_data, list) else stopwords_data['stopwords']


def preprocess_text(text, stopwords):
    text = re.sub(r'\W', ' ', text)
    text = text.lower()
    text = ' '.join([word for word in text.split() if word not in stopwords])
    return text


app = Flask(__name__)
CORS(app)


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    news = data['news']
    processed_news = preprocess_text(news, stopwords)
    news_vector = vectorizer.transform([processed_news])
    prediction = model.predict(news_vector)
    return jsonify({'prediction': 'TRUE' if prediction[0] == 1 else 'FAKE'})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
