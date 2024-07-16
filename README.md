# Fake News Detection Project

## Overview
This project focuses on building a robust fake news detection system using machine learning techniques. The objective is to classify news articles as either "TRUE" or "FAKE" based on their content. The project utilizes various machine learning models, including Logistic Regression, Naive Bayes, and Support Vector Machine (SVM), combined into an ensemble model for improved accuracy and reliability.

## Project Structure
- **Data Preprocessing:** The dataset is preprocessed to remove punctuation, convert text to lowercase, and eliminate stopwords.
- **Feature Extraction:** TF-IDF (Term Frequency-Inverse Document Frequency) is used to convert the text data into numerical features suitable for machine learning models.
- **Model Training:** Three individual models (Logistic Regression, Naive Bayes, and SVM) are trained on the preprocessed data. An ensemble model using Voting Classifier is then created to leverage the strengths of these individual models.
- **Model Evaluation:** The performance of the individual models and the ensemble model is evaluated using accuracy scores and classification reports.
- **Prediction Function:** A function is provided to predict the authenticity of new news articles.

## Files and Directories
- `Combined.csv`: The CSV file containing the dataset of news articles and their labels.
- `stopwords.json`: A JSON file containing the list of stopwords to be removed during text preprocessing.
- `ensemble_model.pkl`: The saved ensemble model for future predictions.
- `tfidf_vectorizer.pkl`: The saved TF-IDF vectorizer used to transform text data.

## Requirements
- Python 3.x
- pandas
- scikit-learn
- json
- re
- pickle

## Usage
1. **Preprocessing Data:**
   ```python
   data['processed_news'] = data['News Items'].apply(lambda x: preprocess_text(str(x), stopwords))
2. **Feature Extraction:**
3. **Model Training:**
   ```python
   vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(data['processed_news'])
  3. **Model Training:**
     ```python
     logistic_regression.fit(X_train, y_train)
     naive_bayes.fit(X_train, y_train)
     svm.fit(X_train, y_train)
     ensemble_model.fit(X_train, y_train)
  4. **Saving Models:**
     ```python
     with open('ensemble_model.pkl', 'wb') as model_file:
     pickle.dump(ensemble_model, model_file)
     with open('tfidf_vectorizer.pkl', 'wb') as vectorizer_file:
     pickle.dump(vectorizer, vectorizer_file)
  
  5. **Predicting News:**
     ```python
     prediction = predict_news(new_news, ensemble_model, vectorizer, stopwords)
     print(f"The news is predicted to be: {prediction}")

  6. **Evaluation:**
      The models are evaluated based on their accuracy and detailed classification reports. The ensemble model generally shows better performance compared to individual models.

     **Example Usage:**
     ```python
      new_news = "سوشل میڈیا پر وائرل مبینہ طور پر خیبرپختونخوا (KP) کے وزیر اعلیٰ علی امین خان گنڈا پور کی اپنی مشیر مشال یوسفزئی کو ایک بالکل نئی لگژری کار کی چابیاں تحفے میں دیتے ہوئے تصویر نے ایک نئی بحث کو جنم دیا ہے۔"
      prediction = predict_news(new_news, ensemble_model, vectorizer, stopwords)
      print(f"The news is predicted to be: {prediction}")
 ## Results
- **Logistic Regression Accuracy:** [Accuracy Score]
- **Naive Bayes Accuracy:** [Accuracy Score]
- **SVM Accuracy:** [Accuracy Score]
- **Ensemble Model Accuracy:** [Accuracy Score]

### Classification Reports:
- **Logistic Regression:** [Classification Report]
- **Naive Bayes:** [Classification Report]
- **SVM:** [Classification Report]
- **Ensemble Model:** [Classification Report]

## Conclusion
This project demonstrates the effectiveness of using an ensemble model for fake news detection, combining the strengths of multiple machine learning algorithms to achieve higher accuracy and reliability.







